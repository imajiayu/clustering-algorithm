from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import math

datas=load_iris()['data']
datas_train=[[x[2],x[3]] for x in datas]
for i in range(len(datas_train)):
    datas_train[i].append(i)

#计算欧氏距离
def distEclud(vecA,vecB):
    return math.sqrt(sum((np.array(vecA)-np.array(vecB))**2))

#计算两个类之间的距离，方法为average
def class_distance(class1,class2):
    total_distance=0
    for elem1 in class1:
        distance=0
        for elem2 in class2:
            distance+=distEclud(elem1[:-1],elem2[:-1])
        total_distance+=distance/len(class2)
    total_distance/=len(class1)
    return total_distance

#返回距离最近的两个类的下标
def merge_arg(classes):
    class1=-1
    class2=-1
    min_distance=math.inf
    for i in range(len(classes)-1):
        for j in range(i+1,len(classes)):
            distance=class_distance(classes[i],classes[j])
            if distance<min_distance:
                min_distance=distance
                class1=i
                class2=j
    return class1,class2

#迭代，直到类的个数为指定值
def HAC(datas_train,class_num):
    classes=[[x] for x in datas_train]
    while len(classes)!=class_num:
        i,j=merge_arg(classes)
        classes[i]+=classes[j]
        classes.pop(j)
    return classes

#画图
result=HAC(datas_train,3)
class1=np.array(result[0])
class2=np.array(result[1])
class3=np.array(result[2])
plt.scatter(class1[:,0],class1[:,1],c='r')
plt.scatter(class2[:,0],class2[:,1],c='g')
plt.scatter(class3[:,0],class3[:,1],c='b')
plt.show()