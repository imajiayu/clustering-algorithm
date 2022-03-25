import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import random
from sklearn.datasets import load_iris

datas_iris=load_iris()
datas_x=np.array(datas_iris['data'])
datas_y=np.array(datas_iris['target'])

datas=np.column_stack((datas_x,datas_y))

class1=np.array([x for x in datas if x[4]==0])
class2=np.array([x for x in datas if x[4]==1])
class3=np.array([x for x in datas if x[4]==2])

#取二维数据作为输入 画出参考分类图
'''
plt.scatter(class1[:,2],class1[:,3],c='r',label='class1')
plt.scatter(class2[:,2],class2[:,3],c='g',label='class2')
plt.scatter(class3[:,2],class3[:,3],c='b',label='class3')
plt.show()
'''
#计算欧氏距离
def distEclud(vecA,vecB):
    return math.sqrt(sum((vecA-vecB)**2))

#取得k个中心，其质心随机
datas_train=np.array([[x[2],x[3]] for x in datas])
def randCent(datas_train,k):
    n=datas_train.shape[1]
    centroids=np.zeros((k,n))
    for j in range(n):
        minJ=min(datas_train[:,j])
        maxJ=max(datas_train[:,j])
        centroids[:,j]=minJ+(maxJ-minJ)*random.rand(1,k)
    return centroids

#K-means 聚类算法
def kMeans(datas_train, k, distMeans =distEclud, createCent = randCent):
    m=datas_train.shape[0]
    clusterAssment=np.zeros((m,2))
    centroids=createCent(datas_train,k)
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        for i in range(m):
            minDist=math.inf
            minIndex=-1
            for j in range(k):
                dist=distMeans(datas_train[i],centroids[j])
                if dist<minDist:
                    minDist=dist
                    minIndex=j
            if clusterAssment[i][0]!=minIndex:
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2
        for cent in range(k):
            ptsInClust=datas_train[np.nonzero(clusterAssment[:,0] == cent)]
            centroids[cent,:]=np.mean(ptsInClust,axis=0)
    return centroids,clusterAssment

def getSSE():
    loss=[]
    for i in range(0,10):
        distSum=np.sum(kMeans(datas_train,i+1)[1][:,1])
        loss.append(distSum)
    print(loss)
    plt.plot([x+1 for x in range(0,10)],loss)
    plt.show()

def draw_result():
    result=kMeans(datas_train,3)[1]
    n=datas_train.shape[0]
    for i in range(n):
        if result[i][0]==0:
            plt.scatter(datas_train[i][0],datas_train[i][1],c='b')
        elif result[i][0]==1:
            plt.scatter(datas_train[i][0],datas_train[i][1],c='g')
        else:
            plt.scatter(datas_train[i][0],datas_train[i][1],c='r')
    plt.show()


getSSE()







