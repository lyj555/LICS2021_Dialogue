# -*- coding: utf-8 -*-

import re
import json


class ToSampleForDataSource(object):
    @staticmethod
    def to_sample_for_douban(input_file, data_type="chitchat", is_test=False):
        """
        base data info: 豆瓣多轮对话	500000(train)	25001(dev)	1186(test)	单轮	开放域
        :param input_file:
        :param data_type:
        :param is_test:
        :return:
        """
        with open(input_file, encoding='utf8') as fp:
            for line in fp:
                data = json.loads(line.strip())

                history = data["history"]
                response = data["response"] if "response" in data else ""

                if not is_test:
                    sample = {"type": data_type,
                              "knowledge": "",
                              "context": history,
                              "response": response}
                    yield sample
                else:
                    sample = {"type": data_type,
                              "knowledge": "",
                              "context": '\t'.join(history),
                              "response": response}

                    yield sample

    @staticmethod
    def to_sample_for_lccc(input_file, data_type="chitchat", is_test=False):
        """
        base data info:	清华LCCC	11987759(train)	20000(dev)	10000(dev)	多轮	开放域
        :param input_file:
        :param data_type:
        :param is_test:
        :return:
        """
        with open(input_file, encoding='utf8') as fp:
            for line in fp:
                data = json.loads(line.strip())
                if not is_test:
                    conversation = data["conversation"]

                    for i in range(1, len(conversation)):
                        sample = {"type": data_type,
                                  "knowledge": "",
                                  "context": '\t'.join(conversation[:i]),
                                  "response": conversation[i]}

                        yield sample
                else:
                    history = data["history"]
                    response = data["response"] if "response" in data else ""

                    sample = {"type": data_type,
                              "knowledge": "",
                              "context": '\t'.join(history),
                              "response": response}

                    yield sample

    @staticmethod
    def to_sample_for_weibo(input_file, data_type="chitchat", is_test=False):
        """
        process weibo data
        :param input_file:
        :param data_type:
        :param is_test:
        :return:
        """
        with open(input_file, encoding='utf8') as fp:
            for line in fp:
                data = json.loads(line.strip())

                history = data["history"]
                response = data["response"] if "response" in data else ""

                if not is_test:
                    sample = {"type": data_type,
                              "knowledge": "",
                              "context": history,
                              "response": response}

                    yield sample

                else:
                    sample = {"type": data_type,
                              "knowledge": "",
                              "context": '\t'.join(history),
                              "response": response}

                    yield sample

    @staticmethod
    def to_sample_for_duconv(input_file, data_type="knowledge", is_test=False):
        with open(input_file, encoding='utf8') as fp:
            for line in fp:
                data = json.loads(line.strip())

                goal = data["goal"]
                knowledge = data["knowledge"]

                goal_knowledge = ' '.join([' '.join(spo) for spo in goal + knowledge])

                if not is_test:
                    conversation = data["conversation"]

                    for i in range(0, len(conversation), 2):
                        sample = {"type": data_type,
                                  "knowledge": goal_knowledge,
                                  "context": '\t'.join(conversation[:i]),
                                  "response": conversation[i]}

                        yield sample
                else:
                    history = data["history"]
                    response = data["response"] if "response" in data else ""

                    sample = {"type": data_type,
                              "knowledge": goal_knowledge,
                              "context": '\t'.join(history),
                              "response": response}

                    yield sample

    @staticmethod
    def to_sample_for_kdconv(input_file, data_type="knowledge", is_test=False):
        with open(input_file, encoding='utf8') as fp:
            for line in fp:
                data = json.loads(line.strip())
                knowledge = data["knowledge"]

                knowledge = ' '.join([' '.join(spo) for spo in knowledge])

                if not is_test:
                    conversation = data["conversation"]
                    for i in range(len(conversation)):
                        sample = {"type": data_type,
                                  "knowledge": knowledge,
                                  "context": '\t'.join(conversation[:i]),
                                  "response": conversation[i]}

                        yield sample
                else:
                    history = data["history"]
                    response = data["response"] if "response" in data else ""

                    sample = {"type": data_type,
                              "knowledge": knowledge,
                              "context": '\t'.join(history),
                              "response": response}

                    yield sample

    @staticmethod
    def to_sample_for_tencent(input_file, data_type="knowledge", is_test=False):
        with open(input_file, encoding='utf8', errors="ignore") as fp:
            for line in fp:
                data = json.loads(line.strip())

                knowledge = data["knowledge"]
                history = data["history"]
                response = data["response"] if "response" in data else ""

                knowledge = ' '.join(knowledge)

                if not is_test:
                    sample = {"type": data_type,
                              "knowledge": knowledge,
                              "context": history,
                              "response": response}

                    yield sample

                else:
                    sample = {"type": data_type,
                              "knowledge": knowledge,
                              "context": '\t'.join(history),
                              "response": response}

                    yield sample

    @staticmethod
    def to_sample_for_durecdial(input_file, data_type="recommend", is_test=False):
        def goal_processing(goal):
            format_goal = []
            while isinstance(goal, list):
                goal = goal[0]
            goal = goal.split('-->')
            for i, g in enumerate(goal):
                format_g = []
                g = g.strip()
                si, ei = g.find('['), g.find(']')
                if si != 0 or ei <= si + 1 or not g[si + 1:ei].isdigit():
                    continue

                g = g.split(g[si:ei + 1])[-1]
                g_n = g.split('(', 1)[0].strip()
                g_d = g.split('(', 1)[-1].strip()

                format_g.append(g_n)

                if "新闻" in g_n or g_n.replace(' ', '') in ["关于明星的聊天", "兴趣点推荐", "音乐推荐", "播放音乐", "美食推荐", "poi推荐", "电影推荐",
                                                           "音乐点播", "问日期", "新闻推荐", "新闻点播", "问答"]:
                    left = -1
                    for right, c in enumerate(g_d):
                        if c == "『":
                            left = right + 1
                        elif c == "』":
                            if 0 <= left < right:
                                item = g_d[left:right].strip()
                                if item not in format_g and item.replace(' ', '') != "参考知识":
                                    format_g.append(item)
                            left = -1

                format_goal.append(format_g)

            if len(format_goal) > 3:
                format_goal = [format_goal[0], format_goal[-2], format_goal[-1]]

            return format_goal

        def user_profile_processing(user_profile):
            accept_key = ["拒绝", "喜欢的电影", "喜欢的明星", "喜欢的poi", "喜欢的音乐",
                          "喜欢的新闻", "同意的新闻", "同意的音乐", "同意的美食", "同意的poi", "同意的电影"]
            format_user_profile = []
            for key in user_profile:
                if key.replace(' ', '') in accept_key:
                    if isinstance(user_profile[key], list):
                        format_user_profile.append([key, ' '.join(user_profile[key])])
                    else:
                        format_user_profile.append([key, user_profile[key]])

            return format_user_profile

        def strip_utterance(utterance_list):
            for i, utterance in enumerate(utterance_list):
                utterance = utterance.split(' ')
                if re.match("\[\d+\]", utterance[0]) is not None:
                    utterance = utterance[1:]
                utterance = ' '.join(utterance)
                utterance_list[i] = utterance

        with open(input_file, encoding='utf8') as fp:
            for line in fp:
                data = json.loads(line.strip())

                situation = data["situation"]
                goal = data["goal"]
                user_profile = data["user_profile"]
                knowledge = data["knowledge"]

                goal = goal_processing(goal)
                user_profile = user_profile_processing(user_profile)

                bot_mode = 0 if goal[0][0] == '寒暄' else 1

                goal = ' '.join([' '.join(g) for g in goal])
                user_profile = ' '.join([' '.join(up) for up in user_profile])
                knowledge = ' '.join([' '.join(spo) for spo in knowledge])

                background = ' '.join([goal, situation, user_profile, knowledge])

                if not is_test:
                    conversation = data["conversation"]

                    strip_utterance(conversation)

                    for i, utterance in enumerate(conversation):
                        if i % 2 != bot_mode:
                            continue

                        sample = {"type": data_type,
                                  "knowledge": background,
                                  "context": '\t'.join(conversation[:i]),
                                  "response": conversation[i]}

                        yield sample
                else:
                    history = data["history"]
                    response = data["response"] if "response" in data else ""

                    strip_utterance(history)

                    sample = {"type": data_type,
                              "knowledge": background,
                              "context": '\t'.join(history),
                              "response": response}

                    yield sample

    @staticmethod
    def to_sample_for_persona(input_file, data_type="persona", is_test=False):
        with open(input_file, encoding='utf8') as fp:
            for line in fp:
                data = json.loads(line.strip())
                if not is_test:
                    conversation = data["conversation"]
                    p1_persona = [u'画像 : ' + p for p in data['p1_persona']]
                    p2_persona = [u'画像 : ' + p for p in data['p2_persona']]
                    for reply_index in range(0, len(conversation)):
                        persona = None
                        if conversation[reply_index].startswith('p1 : '):
                            persona = p1_persona
                        elif conversation[reply_index].startswith('p2 : '):
                            persona = p2_persona
                        else:
                            print(line)
                            print >> sys.stderr, "invalid personal tag"
                            continue
                        conversation[reply_index] = conversation[reply_index].lstrip('p1 : ').lstrip('p2 : ')
                        if reply_index == 0:
                            continue

                        sample = {"type": data_type,
                                  "knowledge": '\t'.join(persona),
                                  "context": '\t'.join(conversation[:reply_index]),
                                  "response": conversation[reply_index]}

                        yield sample
                else:
                    persona = [u'画像 : ' + data["profile"][pkey] for pkey in data["profile"]]

                    history = data["history"]
                    response = data["response"] if "response" in data else ""

                    sample = {"type": data_type,
                              "knowledge": '\t'.join(persona),
                              "context": '\t'.join(history),
                              "response": response}

                    yield sample

    @staticmethod
    def to_sample_for_emotional(input_file, data_type="knowledge", is_test=False):
        emotion_dict = {0: '空', 1: '喜欢', 2: '伤心', 3: '厌恶', 4: '愤怒', 5: '高兴'}
        with open(input_file, encoding='utf8') as fp:
            for line in fp:
                data = json.loads(line.strip())

                knowledge = data["knowledge"]
                history = data["history"]
                response = data["response"] if "response" in data else ""

                emotion = ' '.join([emotion_dict[knowledge[0]], emotion_dict[knowledge[1]]])

                if not is_test:
                    sample = {"type": data_type,
                              "knowledge": emotion,
                              "context": history,
                              "response": response}
                    yield sample
                else:
                    sample = {"type": data_type,
                              "knowledge": emotion,
                              "context": '\t'.join(history),
                              "response": response}
                    yield sample
