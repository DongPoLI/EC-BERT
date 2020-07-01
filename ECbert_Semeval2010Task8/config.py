# -*- coding: utf-8 -*-
"""
@Time   : 2020/7/1
@Author : Li Shenzhen
@File   : config.py
@Software: PyCharm
"""


# SemEval2010_task8 Test Config
class ECbert_Config:
    test_path = "datas/TEST_FILE_FULL.TXT"
    checkpoint_dir = "model/8969checkpoint.pth.tar"
    predictions = "predictions/"
    max_sentence_length = 128
    val_batch_size = 32
    is_cuda = True
    pretrained_weights = "bert-large-uncased"   # BERT Large

    # 文字关系：标签 19;
    class2label = {'Other': 0,
                   'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
                   'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
                   'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
                   'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
                   'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
                   'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
                   'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
                   'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
                   'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}

    # 标签： 文字关系
    label2class = {0: 'Other',
                   1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
                   3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
                   5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
                   7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
                   9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
                   11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
                   13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
                   15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
                   17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}


config = ECbert_Config()

if __name__ == "__main__":

    print(config.label2class)