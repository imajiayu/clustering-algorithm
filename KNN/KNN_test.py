from sklearn.datasets import load_iris,load_digits
import numpy as np
from math import sqrt
from collections import Counter
import sys

f=open('./KNN_test.txt_output','w')
sys.stdout=f

datas_iris=load_iris()#导入数据集
x_data=np.array(datas_iris['data'][0:150])
y_data=np.array(datas_iris['target'][0:150])

datas=np.column_stack((x_data,y_data))#将数据集变为维度为(150,5)的矩阵，第五列表示类别

datas_train=datas[0:150:2]#划分训练集和测试集
datas_test=datas[0:150:2]

def get_distances(x0,datas_train):#计算一个数据与整个训练集的距离，并由小到大排序
    distances=[]
    for x_t in datas_train:
        distant=sqrt(np.sum((x0[0:4]-x_t[0:4])**2))#欧氏距离
        distances.append([distant,x_t[4]])
    return sorted(distances)

def predict(x0,datas_train,k):#根据k值预测x0的类别
    distances=get_distances(x0,datas_train)
    predict_target_list=[distances[i][1] for i in range(0,k)]
    votes=Counter(predict_target_list)#统计前k个数据中每个类别的数量
    return votes.most_common(1)[0][0]#取最多的那个

def get_precision(datas_train,datas_test,k):#遍历测试集，计算准确度
    result=[0,0]
    for x0 in datas_test:
        pre=predict(x0,datas_train,k)
        print("预测值为",x0[4],"真实值为",pre)
        if x0[4] == pre:
            result[0]=result[0]+1
        else:
             result[1]=result[1]+1
    return result[0]/(result[0]+result[1])

def iteration(datas_train,datas_test):#迭代求使得精确度最高的k值
    best_precision=0
    parameter=()
    for k in range(1,75):
        precision=get_precision(datas_train,datas_test,k)
        print("k=",k,"准确度=",precision)
        if precision>best_precision:
            best_precision=precision
            parameter=(k,best_precision)
    return parameter

print("最佳k与其准确率为",iteration(datas_test,datas_train))

f.close()



