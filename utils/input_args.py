# -*- coding: utf-8 -*-

import argparse


def parse_args(input_arg=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--do_train", default=1, type=int, help="Whether to train the model.")
    parser.add_argument('--train_data_path', type=str, default='./datasets/train.txt',
                        help='Specify the path to load train data.')
    parser.add_argument("--do_eval", default=1, type=int, help="Whether or not do evaluation")
    parser.add_argument('--valid_data_path', type=str, default='./datasets/valid.txt',
                        help='Specify the path to load valid data.')
    parser.add_argument("--do_predict", default=1, type=int, help="Whether or not predict")
    parser.add_argument('--test_data_path', type=str, default='./datasets/test.txt',
                        help='Specify the path to load test data.')
    parser.add_argument("--model_type", default="uniLM", type=str,
                        help="Type of pre-trained model, now supported values is [uniLM, ] ")
    parser.add_argument('--pretrained_model_path', type=str, default='unified_transformer-12L-cn',
                        help='The path or shortcut name of the pre-trained model.')
    parser.add_argument("--output_dir", type=str, default="output",
                        help="The output directory where the model predictions and checkpoints will be written.")
    # run environment config
    parser.add_argument('--seed', type=int, default=2021, help='Random seed for initialization.')
    parser.add_argument("--device", type=str, default="gpu", help="Device for selecting for the training.")

    # model config
    parser.add_argument('--train_epochs', type=int, default=3,
                        help='Total number of training epochs to perform.')
    parser.add_argument('--batch_size', type=int, default=4, required=True,
                        help='Batch size per GPU/CPU for training.')
    ## model optimization
    parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='The weight decay for optimizer.')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='The number of warmup steps.')
    parser.add_argument('--max_grad_norm', type=float, default=0.1, help='The max value of grad norm.')

    # model steps
    parser.add_argument('--logging_steps', type=int, default=500, help='Log every X updates steps.')
    parser.add_argument('--save_steps', type=int, default=100000, help='Save checkpoint every X updates steps.')

    # model lengths
    parser.add_argument('--sort_pool_size', type=int, default=65536, help='The pool size for sort in build batch data.')
    parser.add_argument('--min_dec_len', type=int, default=1, help='The minimum sequence length of generation.')
    parser.add_argument('--max_dec_len', type=int, default=64, help='The maximum sequence length of generation.')
    parser.add_argument('--num_samples', type=int, default=20, help='The decode numbers in generation.')
    parser.add_argument('--decode_strategy', type=str, default='sampling', help='The decode strategy in generation.')
    parser.add_argument('--top_k', type=int, default=5,
                        help='The number of highest probability vocabulary tokens to keep for top-k sampling.')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='The value used to module the next token probabilities.')
    parser.add_argument('--top_p', type=float, default=1.0, help='The cumulative probability for top-p sampling.')
    parser.add_argument('--num_beams', type=int, default=30, help='The number of beams for beam search.')
    parser.add_argument('--length_penalty', type=float, default=1.0,
                        help='The exponential penalty to the sequence length for beam search.')
    parser.add_argument('--early_stopping', type=eval, default=True,
                        help='Whether to stop the beam search when at least `num_beams`'
                             ' sentences are finished per batch or not.')

    if input_arg:
        args = parser.parse_args(input_arg.split())
    else:
        args = parser.parse_args()
    return args

