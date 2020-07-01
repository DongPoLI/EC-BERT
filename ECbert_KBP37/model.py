import torch
import torch.nn as nn
import torchvision
from transformers import BertModel, BertPreTrainedModel


class ECbert_Large(BertPreTrainedModel):
    def __init__(self, config, tokenizer, max_sentence_length=128, is_cuda=True, num_labels=37,
                 pretrained_weights="bert-large-uncased"):
        super(ECbert_Large, self).__init__(config)
        self.num_labels = num_labels
        self.max_sentence_length = max_sentence_length  # 最长序列长度

        self.tokenizer = tokenizer
        self.bertModel = BertModel.from_pretrained(pretrained_weights, config=config)  # , config=config
        self.is_cuda = is_cuda

        d = config.hidden_size  # bert-Large-uncased
        self.entity_dense = nn.Linear(d, d)
        self.CLS_dense = nn.Linear(d, d)
        self.ABC_dense = nn.Linear(d, d)
        self.all_dense = nn.Linear(d * 6, self.num_labels)

        # 线型层 参数初始化  有效果
        nn.init.xavier_normal_(self.entity_dense.weight)
        nn.init.constant_(self.entity_dense.bias, 0.)
        nn.init.xavier_normal_(self.CLS_dense.weight)
        nn.init.constant_(self.CLS_dense.bias, 0.)
        nn.init.xavier_normal_(self.all_dense.weight)
        nn.init.constant_(self.all_dense.bias, 0)
        nn.init.xavier_normal_(self.ABC_dense.weight)
        nn.init.constant_(self.ABC_dense.bias, 0)

        self.dorpout = nn.Dropout(0.1)

    def forward(self, x, input_masks, x_mark_index_all, device):

        bertresult, _ = self.bertModel(x, input_masks)

        batch_size = x.size()[0]  # size
        doler_result = []
        jin_result = []
        A_result = []
        B_result = []
        C_result = []
        for i in range(batch_size):
            cls = x_mark_index_all[i][0]
            doler = x_mark_index_all[i][1]
            jin = x_mark_index_all[i][2]
            sep = x_mark_index_all[i][3]

            # $ entity $
            entity1 = torch.mean(bertresult[i, doler[0] + 1: doler[1], :], dim=0, keepdim=True)
            doler_result.append(entity1)

            # # entity #
            entity2 = torch.mean(bertresult[i, jin[0] + 1: jin[1], :], dim=0, keepdim=True)
            jin_result.append(entity2)

            # [CLS] A $
            a_vector = bertresult[i, cls[0]+1: doler[0], :]  # , dim=0, keepdim=True
            if a_vector.size()[0] != 0:
                A_result.append(torch.mean(a_vector, dim=0, keepdim=True))
            else:  # 如果是 [CLS] $ entity $ 情况， 则获取[CLS]
                A_result.append(torch.mean(bertresult[i, cls[0]: doler[0], :], dim=0, keepdim=True))

            # $ B #
            b_vector = bertresult[i, doler[1]+1:jin[0], :]
            if b_vector.size()[0] != 0:
                B_result.append(torch.mean(b_vector, dim=0, keepdim=True))
            else:  # $ entity $ # entity # C [SEP]
                B = (entity1 + entity2) / 2
                B_result.append(B)

            # # C [SEP]
            c_vector = bertresult[i, jin[1]+1:sep[0], :]
            if c_vector.size()[0] != 0:
                C_result.append(torch.mean(c_vector, dim=0, keepdim=True))
            else:
                C_result.append(torch.mean(bertresult[i, sep[0]:sep[0]+1, :], dim=0, keepdim=True))

        # 拼接
        H_clr = bertresult[:, 0]
        H_doler = torch.cat(doler_result, 0)
        H_jin = torch.cat(jin_result, 0)
        H_A = torch.cat(A_result, 0)
        H_B = torch.cat(B_result, 0)
        H_C = torch.cat(C_result, 0)

        cls_dense = self.CLS_dense(self.dorpout(torch.tanh(H_clr)))
        doler_dense = self.entity_dense(self.dorpout(torch.tanh(H_doler)))
        jin_dense = self.entity_dense(self.dorpout(torch.tanh(H_jin)))
        A_dense = self.ABC_dense(self.dorpout(torch.tanh(H_A)))
        B_dense = self.ABC_dense(self.dorpout(torch.tanh(H_B)))
        C_dense = self.ABC_dense(self.dorpout(torch.tanh(H_C)))

        cat_result = torch.cat((cls_dense, A_dense, doler_dense, B_dense, jin_dense, C_dense), 1)

        result = self.all_dense(cat_result)

        return result

