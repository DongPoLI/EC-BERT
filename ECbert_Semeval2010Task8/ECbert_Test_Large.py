# -*- coding: utf-8 -*-
"""
@Time   : 2020/7/1
@Author : Li Shenzhen
@File   : ECbert_Test_Large.py
@Software: PyCharm
"""
import subprocess
import data_helpers
import os
import torch
from model import ECbert_Large
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import f1_score
import numpy as np
from transformers import BertConfig
import torch.nn as nn
from config import config

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置GPU ID
# gpu 设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)


test_path = config.test_path
checkpoint_dir = config.checkpoint_dir
predictions = config.predictions
max_sentence_length = config.max_sentence_length
val_batch_size = config.val_batch_size
is_cuda = config.is_cuda
pretrained_weights = config.pretrained_weights

rt_tokenizer = data_helpers.tokenizer
cls_id = rt_tokenizer.cls_token_id
sep_id = rt_tokenizer.sep_token_id
pad_id = rt_tokenizer.pad_token_id
doler_id, jin_id = rt_tokenizer.additional_special_tokens_ids


bert_config = BertConfig.from_pretrained(pretrained_weights)
bert_config.num_labels = 19
model = ECbert_Large(bert_config, rt_tokenizer, max_sentence_length=max_sentence_length, is_cuda=is_cuda, pretrained_weights=pretrained_weights)
if torch.cuda.device_count() > 1:  # 使用多GPU
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

model.to(device)  # model给到 GPU


# 数据
class RCDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


def eval():
    x_val, y_val, _ = data_helpers.load_data_and_labels(test_path)

    # 加载模型
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    val_dataset = RCDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    pred_y = []
    labels = []
    with torch.no_grad():
        for index, (val_text, val_label) in enumerate(val_loader):
            labels.extend(val_label.tolist())
            # 数据转化为 Token Id
            x_token = []
            x_mark_index_all = []
            for item in val_text:
                temp = rt_tokenizer.encode(item, add_special_tokens=False)
                while len(temp) < max_sentence_length:  # 所有序列 填充 PAD 一样长度
                    temp.append(pad_id)
                # 确定 CLS $ $ # # 对应的坐标; 记住 对应的下标
                temp_cup = list(enumerate(temp))
                cls_index = [index for index, value in temp_cup if value == cls_id]  # CLS 都是第一个
                cls_index.append(0)
                doler_index = [index for index, value in temp_cup if value == doler_id]  # $ 两个
                jin_index = [index for index, value in temp_cup if value == jin_id]  # # 两个
                sep_index = [index for index, value in temp_cup if value == sep_id]
                sep_index.append(0)
                x_mark_index = []
                x_mark_index.append(cls_index)
                x_mark_index.append(doler_index)  # 获取两个索引之间的所有数据
                x_mark_index.append(jin_index)  # 获取两个索引之间的所有数据
                x_mark_index.append(sep_index)
                x_mark_index_all.append(x_mark_index)
                x_token.append(temp)

            x_token = np.array(x_token)  # 序列长度一致了
            x_token = torch.tensor(x_token)
            x_token.to(device)

            out = model(x_token, x_mark_index_all, device)
            pred_y.extend(torch.max(out, 1)[1].tolist())  # 预测的标签存储下来

        f1_value = f1_score(labels, pred_y, average='macro')
        val_acc = np.mean(np.equal(labels, pred_y))
        print("Test(非官方): ACC: {}, F1: {}".format(val_acc, f1_value))

        prediction_path = os.path.join(predictions, "predictions.txt")
        truth_path = os.path.join(predictions, "ground_truths.txt")
        prediction_file = open(prediction_path, 'w')
        truth_file = open(truth_path, 'w')
        for i in range(len(pred_y)):
            prediction_file.write("{}\t{}\n".format(i, config.label2class[pred_y[i]]))
            truth_file.write("{}\t{}\n".format(i, config.label2class[labels[i]]))
        prediction_file.close()
        truth_file.close()
        # perl语言文件的源程序
        perl_path = os.path.join("datas",
                                 "semeval2010_task8_scorer-v1.2.pl")
        process = subprocess.Popen(["perl", perl_path, prediction_path, truth_path], stdout=subprocess.PIPE)
        # test = process.communicate()[0].decode("utf-8")
        for line in str(process.communicate()[0].decode("utf-8")).split("\\n"):
            print(line)


if __name__ == "__main__":
    eval()