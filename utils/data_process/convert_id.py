# -*- coding: utf-8 -*-

import re


type_dict = {"chitchat": 30001, "knowledge": 30002, "persona": 30002, "recommend": 30003}

# 0: both(context and respone) meet the requirements
# 1: last utterance of context is too long and trunc
# 2: first utterance of context is too long and trunc
# 3: other utterance of context is too long and trunc
# 4: respone is too long  and trunc
# truncate_type_stat = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}


def truncate_ids_list(ids_list, cut_len=512, truncate_first_turn=False):
    if sum([len(x) for x in ids_list]) <= cut_len:
        return 0, ids_list

    new_ids_list = []
    ids_list.reverse()
    len_cnt = 0
    cut_type = 0

    for i, ids in enumerate(ids_list):
        if len_cnt + len(ids) > cut_len:
            if len_cnt == 0 and (len(ids_list) > 1 or not truncate_first_turn):
                new_ids_list.append(ids[-cut_len:])
                len_cnt = cut_len
                cut_type = 1  # last utterance of context is too long
            elif truncate_first_turn and i == len(ids_list) - 1 and len_cnt + 1 < cut_len:
                new_ids_list.append(ids[:cut_len - len_cnt - 1] + [ids[-1]])
                len_cnt = cut_len
                cut_type = 2  # first utterance of context is too long and trunc
            else:
                cut_type = 3  # other utterance of context is too long and trunc
            break
        else:
            len_cnt += len(ids)
            new_ids_list.append(ids)

    new_ids_list.reverse()
    return cut_type, new_ids_list


def convert_sample_to_numerical(input_data, max_seq_len=512,
                                max_response_len=128, truncate_first_turn=False,
                                is_test=False, truncate_type_stat=None, sp=None):
    assert "type" in input_data and "context" in input_data and "response" in input_data and "knowledge" in input_data

    for key in input_data:
        input_data[key] = re.sub("  +", " ", input_data[key])

    data_type = input_data["type"]
    context = input_data["context"]
    response = input_data["response"]
    knowledge = input_data["knowledge"]

    # type
    assert data_type in type_dict
    type_id = type_dict[data_type]

    # tokenize response
    response_ids = sp.EncodeAsIds(response) + [2]
    if len(response_ids) > max_response_len - 1:
        new_response_ids = response_ids[1 - max_response_len:]
        truncate_type_stat[4] += 1
        if not is_test:
            return None
    else:
        new_response_ids = response_ids[:]

    # tokenize context
    context_ids_list = []
    if knowledge != "":
        knowledge_ids = sp.EncodeAsIds(knowledge) + [2]
        context_ids_list.append(knowledge_ids)

    if context != "":
        for utterance in context.split('\t'):
            utterance_ids = sp.EncodeAsIds(utterance) + [2]
            context_ids_list.append(utterance_ids)

    truncate_type, new_context_ids_list = \
        truncate_ids_list(context_ids_list, max_seq_len - max_response_len - 2,
                          truncate_first_turn=truncate_first_turn)
    truncate_type_stat[truncate_type] += 1

    if truncate_type == 1 and not is_test:
        return None

    # deal context tokens
    token_ids = [1, type_id]
    sent_ids = [0, 0]
    for ids in new_context_ids_list:
        token_ids += ids
        sent_ids += ([0] * len(ids))

    # deal reply tokens
    if not is_test:
        token_ids += [1]
        sent_ids += [1]
        token_ids += new_response_ids
        sent_ids += ([1] * len(new_response_ids))

    assert (len(token_ids) == len(sent_ids))
    position_ids = range(len(token_ids))

    output_list = []
    for l in [token_ids, sent_ids, position_ids]:
        output_list.append(' '.join([str(x) for x in l]))

    return output_list


