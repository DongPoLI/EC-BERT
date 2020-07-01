import numpy as np
import pandas as pd
import re
from config import config
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(config.pretrained_weights)
special_tokens_dict = {'additional_special_tokens': ["$", "#"]}
# print(tokenizer.SPECIAL_TOKENS_ATTRIBUTES)
# 添加特殊Token, 使模型不会拆分， 用作标记使用
tokenizer.add_special_tokens(special_tokens_dict)
# print(tokenizer.additional_special_tokens)
# print(tokenizer.additional_special_tokens_ids)
# print(tokenizer.sep_token)
# print(tokenizer.sep_token_id)
# print(tokenizer.cls_token_id)
# print(tokenizer.cls_token)
# print(tokenizer.pad_token_id)
# print(tokenizer.mask_token)  # [MASK]
# print(tokenizer.mask_token_id)  # 103


def clean_str(text):
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=$#]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)  # ?
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)  # ?

    return text.strip()


def load_data_and_labels(path):
    data = []
    lines = [line.strip() for line in open(path)]
    print("数据总长度:{}".format(len(lines) / 4))
    max_sentence_length = 0
    for idx in range(0, len(lines), 4):  # 处理每条句子
        id = lines[idx].split("\t")[0]
        relation = lines[idx + 1]

        sentence = lines[idx].split("\t")[1][1:-1]
        # 清除 原有的 # $, 特殊符号 不认为影响 句子意思
        sentence = sentence.replace('#', '')
        sentence = sentence.replace('$', '')

        sentence = sentence.replace('<e1>', ' $ ')
        sentence = sentence.replace('</e1>', ' $ ')
        sentence = sentence.replace('<e2>', ' # ')
        sentence = sentence.replace('</e2>', ' # ')

        sentence = clean_str(sentence)  # 对句子清洗一遍
        sentence = "[CLS] " + sentence + " [SEP]"  # 在句子开始 加入[CLS] or CLS ？ [CLS]:101; CLS:101

        if "# #" in sentence or "$ $" in sentence:  # 如果 有实体为 空格 则丢弃
            continue
        # tokens = nltk.word_tokenize(sentence)
        tokens = tokenizer.tokenize(sentence)
        if max_sentence_length < len(tokens):
            max_sentence_length = len(tokens)

        data.append([id, sentence, relation])

    # print("最长序列长度为: {}".format(max_sentence_length))
    df = pd.DataFrame(data=data, columns=["id", "sentence", "relation"])
    df['label'] = [config.class2label[r] for r in df['relation']]
    x_text = df['sentence'].tolist()
    y = df['label'].tolist()

    x_text = np.array(x_text)
    y = np.array(y)
    return x_text, y, max_sentence_length  # 数据（句子），标签


if __name__ == "__main__":
    train_path = "datas/kbp37/test.txt"  # 250
    train_path = "datas/kbp37/dev.txt"  # 270
    train_path = "datas/kbp37/train.txt"  # 330 -> 350
    load_data_and_labels(train_path)