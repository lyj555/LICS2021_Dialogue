## 三个子任务，9个数据集。具体为：


1.知识对话数据：百度的DuConv[1]
2.推荐对话数据：百度的DuRecDial[2]
3.画像对话数据：百度的画像对话数据集(Chinese Persona Chat，CPC)
4.其他对话数据：华为的微博数据[3]，北航和微软的豆瓣多轮对话[4]，清华的LCCC[5]，清华的情感对话数据集[6]，清华的KdConv[7]，腾讯的检索辅助生成对话数据集[8]


文献引用
[1] Wenquan Wu, Zhen Guo, Xiangyang Zhou, Hua Wu, Xiyuan Zhang, Rongzhong  Lian, and Haifeng Wang. 2019. Proactive human-machine conversation with  explicit conversation goal. In ACL.
[2] Zeming Liu, Haifeng Wang, Zheng-Yu Niu, Hua Wu, Wanxiang Che, Ting Liu.  2020. Towards Conversational Recommendation over Multi-Type Dialogs. In  ACL.
[3] Lifeng Shang, Zhengdong Lu, Hang Li. 2015. Neural Responding Machine for Short-Text Conversation. In ACL.
[4] Yu Wu, Wei Wu, Chen Xing, Ming Zhou, Zhoujun Li. 2017. Sequential  Matching Network: A New Archtechture for Multi-turn Response Selection  in Retrieval-based Chatbots. In ACL.
[5] Yida Wang, Pei Ke, Yinhe Zheng, Kaili Huang, Yong Jiang, Xiaoyan Zhu, Minlie Huang. 2020. A Large-Scale Chinese Short-Text Conversation Dataset. In NLPCC
[6]Hao Zhou, Minlie Huang, Tianyang Zhang, Xiaoyan Zhu, Bing Liu. 2019. Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory. In AAAI.
[7] Hao Zhou, Chujie Zheng, Kaili Huang, Minlie Huang, Xiaoyan Zhu. 2020.  KdConv: A Chinese Multi-domain Dialogue Dataset Towards Multi-turn  Knowledge-driven Conversation. In ACL.
[8] Deng Cai, Yan Wang, Wei Bi, Zhaopeng Tu, Xiaojiang Liu, Shuming Shi.  2019. Retrieval-guided Dialogue Response Generation via a  Matching-to-Generation Framework. In EMNLP.


## 文件说明
.
|-- Dialog_sample  样例数据集 
|   |-- douban_sample.txt    北航和微软的豆瓣多轮对话
|   |-- duconv_sample.txt    百度的DuConv
|   |-- DuRecDial_sample.txt 百度的DuRecDial
|   |-- Emotional_sample.txt 清华的情感对话数据集
|   |-- kdconv_sample.txt    清华的KdConv
|   |-- LCCC_sample.txt      清华的LCCC
|   |-- Persona_sample.txt   百度的画像对话数据集(Chinese Persona Chat)
|   |-- tencent_sample.txt   腾讯的检索辅助生成对话数据集
|   |-- weibo_sample.txt     华为的微博数据
|   |-- README.md            说明文件
|   `-- License.docx    License

|-- Dialog_train  训练数据集 
|   |-- douban_train.txt    北航和微软的豆瓣多轮对话
|   |-- duconv_train.txt    百度的DuConv
|   |-- DuRecDial_train.txt 百度的DuRecDial
|   |-- Emotional_train.txt 清华的情感对话数据集
|   |-- kdconv_train.txt    清华的KdConv
|   |-- LCCC_train.json      清华的LCCC
|   |-- Persona_train.json   百度的画像对话数据集(Chinese Persona Chat)
|   |-- tencent_train.txt   腾讯的检索辅助生成对话数据集
|   |-- weibo_train.txt     华为的微博数据
|   |-- README.md            说明文件
|   `-- License.docx    License

|-- Dialog_dev    验证集
|   |-- douban_dev.txt    北航和微软的豆瓣多轮对话
|   |-- duconv_dev.txt    百度的DuConv
|   |-- DuRecDial_dev.txt 百度的DuRecDial
|   |-- Emotional_dev.txt 清华的情感对话数据集
|   |-- kdconv_dev.txt    清华的KdConv
|   |-- LCCC_dev.json      清华的LCCC
|   |-- Persona_dev.json   百度的画像对话数据集(Chinese Persona Chat)
|   |-- tencent_dev.txt   腾讯的检索辅助生成对话数据集
|   |-- weibo_dev.txt     华为的微博数据
|   |-- README.md            说明文件
|   `-- License.docx    License

|-- Dialog_testA    测试集A
|   |-- douban_testA.txt    北航和微软的豆瓣多轮对话
|   |-- duconv_testA.txt    百度的DuConv
|   |-- DuRecDial_testA.txt 百度的DuRecDial
|   |-- Emotional_testA.txt 清华的情感对话数据集
|   |-- kdconv_testA.txt    清华的KdConv
|   |-- LCCC_dev.txt      清华的LCCC
|   |-- Persona_testA.json   百度的画像对话数据集(Chinese Persona Chat)
|   |-- tencent_testA.txt   腾讯的检索辅助生成对话数据集
|   |-- weibo_testA.txt     华为的微博数据
|   |-- README.md            说明文件
|   `-- License.docx    License
