# EC-BERT
论文： **Relation Classificaton based on information enhanced BERT**

    EC-BERT 使用 bert-large-uncased   
    在 SemEval-2010 Task 8 Dataset 官方测试 macro-averaged F1 = 89.69%   
    在 KBP37测试集上 macro F1 = 65.92%

## Model Architecture
![](images/ECBERT.png)

## Method
1 从BERT的输出得到6个向量
- [CLS] Token vector
- [CLS] 和 entity_1之前 向量的平均值
- averaged entity_1 vector
- entity_1 和 entity_2 之间的向量的平均值
- averaged entity_2 vector
- entity_2 和 [SEP] 之间向量的平均值

2 Pass each vector to the Dense layers
- tanh -> dropout -> Dense

3 Concatenate 6 vectors.

4 Pass the concatenated vector to Dense layer.

    备注：
    对于文本中 实体之间相邻，有几种特殊情况，处理方法，请看代码
    实体平均 没有把特殊标记包括在内！

## Dependencies

- python >= 3.6
- torch >=  1.4.0+cu92
- transformers == 2.8.0

## RUN
1  SemEval-2010 Task 8 Dataset 上官方测试：
    
    下载模型：https://pan.baidu.com/s/1TqnlXjN3n1cJweiv6OvjBQ  密码: aqu5
    将模型放到 ECbert_Semeval2010Task8/model 文件夹中
    
    cd ECbert_Semeval2010Task8/
    python ECbert_Test_Large.py
 
运行结果
    
    Micro-averaged result (excluding Other):
    P = 2071/2332 =  88.81%     R = 2071/2263 =  91.52%     F1 =  90.14%
    MACRO-averaged result (excluding Other):
    P =  88.32%     R =  91.18%     F1 =  89.69%
    <<< The official score is (9+1)-way evaluation with directionality taken into account: macro-averaged F1 = 89.69% >>>


2   在KBP37数据集上测试:
     
     下载模型：https://pan.baidu.com/s/1RSjnQ5Lh4Gnpg_s1Ci-L1A  密码: 0t94
     放到 ECbert_KBP37/model 文件夹中
     
     cd ECbert_KBP37/
     python ECbert_KBP37_Test.py 


## References

[Semeval 2010 Task 8 Dataset](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?sort=name&layout=list&num=50)  
[KBP37 Dataset](https://github.com/zhangdongxu/kbp37)    
[Huggingface Transformers](https://github.com/huggingface/transformers)  
[R-BERT](https://github.com/monologg/R-BERT)  
[Mul-BERT](https://github.com/DongPoLI/Mul-BERT)


  












