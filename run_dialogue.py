# -*- coding: utf-8 -*-

import os
import random
import time
import math
import logging

import numpy as np
import paddle

import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
from paddle.optimizer.lr import NoamDecay
from paddle.optimizer import AdamW

from paddlenlp.transformers import UnifiedTransformerLMHeadModel, UnifiedTransformerTokenizer

from utils.input_args import parse_args
from utils.data_helper import DialogueDataset, select_response

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)


class ModelOperation(object):
    """ModelTrain"""

    def __init__(self):
        self.cur_process_num = paddle.distributed.get_world_size()  # PADDLE_TRAINERS_NUM 的值，默认值为1
        self.cur_process_rank = paddle.distributed.get_rank()  # PADDLE_TRAINER_ID 的值，默认值为0
        self.model_class = {
            "uniLM": (UnifiedTransformerLMHeadModel, UnifiedTransformerTokenizer),
        }
        self.data_helper = None

    def _initialize_run_env(self, device, seed):
        assert device in ("cpu", "gpu", "xpu"), \
            f"param device({device}) must be in ('cpu', 'gpu', 'xpu')!!!"
        paddle.set_device(device)
        if self.cur_process_num > 1:
            paddle.distributed.init_parallel_env()
        if seed:
            self.set_seed(seed)

    def _initialize_model(self, model_type, pretrained_model_path):
        assert os.path.exists(pretrained_model_path), \
            f"model path {pretrained_model_path} must exists!!!"
        logging.info(f"initialize model from {pretrained_model_path}")

        model_class, tokenizer_class = self.model_class[model_type]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_model_path)
        self.model = model_class.from_pretrained(pretrained_model_path)

        if self.cur_process_num > 1:
            self.model = paddle.DataParallel(self.model)

    def _initialize_optimizer(self, args):
        self.lr_scheduler = NoamDecay(1 / (args.warmup_steps * (args.lr ** 2)),
                                      args.warmup_steps)
        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        self.optimizer = AdamW(
            learning_rate=self.lr_scheduler,
            parameters=self.model.parameters(),
            weight_decay=args.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params,
            grad_clip=nn.ClipGradByGlobalNorm(args.max_grad_norm))

    def _start_train(self, args):
        # load train data loader
        train_dataset = DialogueDataset(
            args.train_data_path,
            args.batch_size,
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            args.sort_pool_size,
            args.seed,
            mode='train')
        train_data_loader = DataLoader(train_dataset, return_list=True, batch_size=None)
        # initialize optimizer
        self._initialize_optimizer(args)
        global_step = 0
        tic_train = time.time()
        for epoch in range(args.train_epochs):
            step = 0
            for batch in train_data_loader:
                logging.info(f"Epoch: {epoch+1}/{args.train_epochs}, step is {step}")
                step += 1
                global_step += 1
                token_ids, type_ids, pos_ids, generation_mask, tgt_label, tgt_pos = batch

                logits = self.model(token_ids, type_ids, pos_ids, generation_mask, tgt_pos)
                loss = F.cross_entropy(logits, tgt_label)

                if global_step % args.logging_steps == 0:
                    logging.info(f"global step {global_step}, epoch: {epoch}, batch: {step},"
                                 f" loss: {loss}, speed: {args.logging_steps / (time.time() - tic_train):.2f} step/s")
                    tic_train = time.time()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.clear_gradients()

        if self.cur_process_rank == 0:
            output_dir = \
                os.path.join(args.output_dir, "model_{}".format(global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # need better way to get inner model of DataParallel
            model_to_save = \
                self.model._layers if isinstance(self.model, paddle.DataParallel) else self.model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print('Saving checkpoint to:', output_dir)

    @paddle.no_grad()
    def evaluation(self, args):
        self.model.eval()
        valid_dataset = DialogueDataset(
            args.valid_data_path,
            args.batch_size,
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            args.sort_pool_size,
            args.seed,
            mode='valid')
        valid_data_loader = DataLoader(valid_dataset, return_list=True, batch_size=None)
        total_tokens = 0
        total_loss = 0.0
        start_time = time.time()
        step = 0
        for inputs in valid_data_loader:
            step += 1
            token_ids, type_ids, pos_ids, generation_mask, tgt_label, tgt_pos = inputs

            logits = self.model(token_ids, type_ids, pos_ids, generation_mask, tgt_pos)
            loss = F.cross_entropy(logits, tgt_label, reduction='sum')

            total_loss += loss.numpy()[0]
            total_tokens += tgt_label.shape[0]

        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        avg_speed = (time.time() - start_time) / step
        logging.info('loss: %.4f - ppl: %.4f - %.3fs/step\n' % (avg_loss, ppl, avg_speed))
        self.model.train()

    @paddle.no_grad()
    def _infer(self, data_loader):
        self.model.eval()
        total_time = 0.0
        start_time = time.time()
        responses = []
        for step, inputs in enumerate(data_loader, 1):
            logging.info(f"step is {step}")
            token_ids, type_ids, pos_ids, generation_mask = inputs
            ids, scores = self.model.generate(
                input_ids=token_ids,
                token_type_ids=type_ids,
                position_ids=pos_ids,
                attention_mask=generation_mask,
                max_length=args.max_dec_len,
                min_length=args.min_dec_len,
                decode_strategy=args.decode_strategy,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                early_stopping=args.early_stopping,
                num_return_sequences=args.num_samples)

            total_time += (time.time() - start_time)
            if step % args.logging_steps == 0:
                logging.info(f'step {step} - {total_time / args.logging_steps:.3f}s/step')
                total_time = 0.0
            results = select_response(ids, scores, self.tokenizer,
                                      args.max_dec_len, args.num_samples)
            responses.extend(results)
            start_time = time.time()
        self.model.train()
        return responses

    def predict(self, args):
        # [1]. initialize dataset loader
        test_dataset = DialogueDataset(
            args.test_data_path,
            args.batch_size,
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            args.sort_pool_size,
            args.seed,
            mode='test')
        valid_data_loader = DataLoader(test_dataset, return_list=True, batch_size=None)
        # [2]. do inference
        responses = self._infer(valid_data_loader)
        # [3]. save result
        output_path = os.path.join(args.output_dir, "predict.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            for response in responses:
                f.write(response + '\n')

    def train_and_eval(self, args):
        self._initialize_run_env(args.device, args.seed)
        self._initialize_model(args.model_type, args.pretrained_model_path)

        # start training
        if args.do_train:
            logging.info("start training...")
            self._start_train(args)
            logging.info("train success.")
        # start evaluation
        if args.do_eval:
            logging.info("start evaluating...")
            self.evaluation(args)
            logging.info("evaluate success.")
        # start predicting
        if args.do_predict:
            logging.info("start predicting...")
            self.predict(args)
            logging.info("predict success.")

    @staticmethod
    def set_seed(random_seed):
        random.seed(random_seed)
        np.random.seed(random_seed)
        paddle.seed(random_seed)


if __name__ == '__main__':
    # input_args = "--do_train 1 --train_data_path ./datasets/output/train.txt " \
    #              "--do_eval 1 --valid_data_path ./datasets/output/train.txt " \
    #              "--do_predict 0 --test_data_path ./datasets/small_test.json " \
    #              "--device cpu --model_type uniLM " \
    #              "--pretrained_model_path unified_transformer-12L-cn --train_epochs 1 " \
    #              "--batch_size 8192"
    args = parse_args(input_arg=None)
    logging.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        logging.info(f'{arg}: {value}')
    logging.info('------------------------------------------------')

    model_oper = ModelOperation()
    model_oper.train_and_eval(args)
