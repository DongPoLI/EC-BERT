# -*- coding: utf-8 -*-
"""
@Time   : 2020/7/1 下午1:57
@Author : Li Shenzhen
@File   : config.py
@Software:PyCharm
"""


class ECbert_Config:
    test_path = "datas/kbp37/test.txt"
    checkpoint_dir = "model/checkpoint.pth.tar"
    predictions = "./predictions/"
    max_sentence_length = 332  # 178 223  -> 250
    val_batch_size = 32
    is_cuda = True
    pretrained_weights = "bert-large-uncased"

    # 文字关系：标签 37;
    class2label = {'no_relation': 0,
                   'org:stateorprovince_of_headquarters(e1,e2)': 1, 'org:stateorprovince_of_headquarters(e2,e1)': 2,
                   'org:alternate_names(e1,e2)': 3, 'org:alternate_names(e2,e1)': 4,
                   'per:cities_of_residence(e1,e2)': 5, 'per:cities_of_residence(e2,e1)': 6,
                   'org:members(e1,e2)': 7, 'org:members(e2,e1)': 8,
                   'per:alternate_names(e1,e2)': 9, 'per:alternate_names(e2,e1)': 10,
                   'org:subsidiaries(e1,e2)': 11, 'org:subsidiaries(e2,e1)': 12,
                   'per:spouse(e1,e2)': 13, 'per:spouse(e2,e1)': 14,
                   'per:countries_of_residence(e1,e2)': 15, 'per:countries_of_residence(e2,e1)': 16,
                   'per:stateorprovinces_of_residence(e1,e2)': 17, 'per:stateorprovinces_of_residence(e2,e1)': 18,
                   'per:employee_of(e1,e2)': 19, 'per:employee_of(e2,e1)': 20,
                   'org:country_of_headquarters(e1,e2)': 21, 'org:country_of_headquarters(e2,e1)': 22,
                   'per:origin(e1,e2)': 23, 'per:origin(e2,e1)': 24,
                   'org:city_of_headquarters(e1,e2)': 25, 'org:city_of_headquarters(e2,e1)': 26,
                   'per:title(e1,e2)': 27, 'per:title(e2,e1)': 28,
                   'org:founded(e1,e2)': 29, 'org:founded(e2,e1)': 30,
                   'org:top_members/employees(e1,e2)': 31, 'org:top_members/employees(e2,e1)': 32,
                   'org:founded_by(e1,e2)': 33, 'org:founded_by(e2,e1)': 34,
                   'per:country_of_birth(e1,e2)': 35, 'per:country_of_birth(e2,e1)': 36,
                   }

    # 标签： 文字关系
    label2class = {0: 'no_relation',
                   1: 'org:stateorprovince_of_headquarters(e1,e2)', 2: 'org:stateorprovince_of_headquarters(e2,e1)',
                   3: 'org:alternate_names(e1,e2)', 4: 'org:alternate_names(e2,e1)',
                   5: 'per:cities_of_residence(e1,e2)', 6: 'per:cities_of_residence(e2,e1)',
                   7: 'org:members(e1,e2)', 8: 'org:members(e2,e1)',
                   9: 'per:alternate_names(e1,e2)', 10: 'per:alternate_names(e2,e1)',
                   11: 'org:subsidiaries(e1,e2)', 12: 'org:subsidiaries(e2,e1)',
                   13: 'per:spouse(e1,e2)', 14: 'per:spouse(e2,e1)',
                   15: 'per:countries_of_residence(e1,e2)', 16: 'per:countries_of_residence(e2,e1)',
                   17: 'per:stateorprovinces_of_residence(e1,e2)', 18: 'per:stateorprovinces_of_residence(e2,e1)',
                   19: 'per:employee_of(e1,e2)', 20: 'per:employee_of(e2,e1)',
                   21: 'org:country_of_headquarters(e1,e2)', 22: 'org:country_of_headquarters(e2,e1)',
                   23: 'per:origin(e1,e2)', 24: 'per:origin(e2,e1)',
                   25: 'org:city_of_headquarters(e1,e2)', 26: 'org:city_of_headquarters(e2,e1)',
                   27: 'per:title(e1,e2)', 28: 'per:title(e2,e1)',
                   29: 'org:founded(e1,e2)', 30: 'org:founded(e2,e1)',
                   31: 'org:top_members/employees(e1,e2)', 32: 'org:top_members/employees(e2,e1)',
                   33: 'org:founded_by(e1,e2)', 34: 'org:founded_by(e2,e1)',
                   35: 'per:country_of_birth(e1,e2)', 36: 'per:country_of_birth(e2,e1)',
                   }


config = ECbert_Config()


if __name__ == "__main__":
    pass