# -*- coding: utf-8 -*-

import os
import argparse

import sentencepiece as spm

from utils.data_process.to_sample_for_data_source import ToSampleForDataSource as tda
from utils.data_process.convert_id import convert_sample_to_numerical


def process_file(data_list):
    for [input_list, output_file, sample_num_threshold] in data_list:
        truncate_type_stat = {i: 0 for i in range(5)}
        with open(output_file, "w") as fout:
            for [input_file, handle_method, truncate_first_turn, is_test] in input_list:
                train_sample_num = 0
                for sample in handle_method(input_file, is_test=is_test):
                    numerical = \
                        convert_sample_to_numerical(sample, is_test=is_test, truncate_type_stat=truncate_type_stat,
                                                    truncate_first_turn=truncate_first_turn, sp=sp)
                    if numerical is not None:
                        train_sample_num += 1
                        fout.write(';'.join(numerical) + "\n")
                    if not is_test and 0 < sample_num_threshold <= train_sample_num:
                        break
        t_sum = sum(truncate_type_stat.values())
        print(f"Total num: {t_sum}")
        for i in range(1, 5):
            print(f"truncate type {i}, num is {truncate_type_stat[i]}, rate is {truncate_type_stat[i]/t_sum}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sp_data_path", default="datasets/spm.model",
                        type=str, help="Whether to train the model.")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.sp_data_path)

    # change the input and output files to your real files
    data_process_list = [
        # [
        #     [
        #         ["./datasets/Dialog_sample/duconv_sample.txt", tda.to_sample_for_duconv, True, False],
        #     ],
        #     "./datasets/output/train_sample.txt",
        #     500
        # ]
        [
            [
                # ["./datasets/Dialog_train/weibo_train.txt", tda.to_sample_for_weibo, False, False],
                # ["./datasets/Dialog_train/douban_train.txt", tda.to_sample_for_douban, False, False],
                # ["./datasets/Dialog_train/LCCC_train.json", tda.to_sample_for_lccc, False, False],
                ["./datasets/Dialog_train/duconv_train.txt", tda.to_sample_for_duconv, True, False],
                # ["./datasets/Dialog_train/kdconv_train.txt", tda.to_sample_for_kdconv, True, False],
                # ["./datasets/Dialog_train/tencent_train.txt", tda.to_sample_for_tencent, True, False],
                ["./datasets/Dialog_train/DuRecDial_train.txt", tda.to_sample_for_durecdial, True, False],
                ["./datasets/Dialog_train/Persona_train.json", tda.to_sample_for_persona, True, False],
                # ["./datasets/Dialog_train/Emotional_train.txt", tda.to_sample_for_emotional, True, False]
            ],
            "./datasets/output/train.txt",
            100000,
        ],
        [
            [
                ["./datasets/Dialog_dev/weibo_dev.txt", tda.to_sample_for_weibo, False, False],
                ["./datasets/Dialog_dev/douban_dev.txt", tda.to_sample_for_douban, False, False],
                ["./datasets/Dialog_dev/LCCC_dev.json", tda.to_sample_for_lccc, False, False],
                ["./datasets/Dialog_dev/duconv_dev.txt", tda.to_sample_for_duconv, True, False],
                # ["./datasets/Dialog_dev/kdconv_dev.txt", tda.to_sample_for_kdconv, True, False],
                # ["./datasets/Dialog_dev/tencent_dev.txt", tda.to_sample_for_tencent, True, False],
                ["./datasets/Dialog_dev/DuRecDial_dev.txt", tda.to_sample_for_durecdial, True, False],
                ["./datasets/Dialog_dev/Persona_dev.json", tda.to_sample_for_persona, True, False],
                # ["./datasets/Dialog_dev/Emotional_dev.txt", tda.to_sample_for_emotional, True, False]
            ],
            "./datasets/output/dev.txt",
            10000,
        ],
        [
            [
                ["./datasets/Dialog_testA/duconv_test.txt", tda.to_sample_for_duconv, True, True],
                ["./datasets/Dialog_testA/DuRecDial_test.txt", tda.to_sample_for_durecdial, True, True],
                ["./datasets/Dialog_testA/Persona_test.json", tda.to_sample_for_persona, True, True],
            ],
            "./datasets/output/test1.txt",
            0
        ],
        [
            [
                ["./datasets/Dialog_testB/duconv_testB.txt", tda.to_sample_for_duconv, True, True],
                ["./datasets/Dialog_testB/DuRecDial_testB.txt", tda.to_sample_for_durecdial, True, True],
                ["./datasets/Dialog_testB/Persona_testB.json", tda.to_sample_for_persona, True, True],
            ],
            "./datasets/output/test2.txt",
            0
        ],
    ]
    process_file(data_process_list)
