import os
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

path='./data/imgs'
fileList=os.listdir(path)

def normalize_image(matrix):#标准化
    mean=np.mean(matrix)
    var=np.mean((matrix-mean)**2)
    image=(matrix-mean)/math.sqrt(var)
    return image

def get_image(filename):#获得标准化后的灰度图片矩阵
    pli_im=Image.open('./data/imgs/{}'.format(filename))
    temp=pli_im.convert('L')
    matrix=np.asarray(temp) 

    matrix=normalize_image(matrix)

    datas_train=matrix.reshape(matrix.shape[0]*matrix.shape[1],1)

    return datas_train,matrix.shape


def randCent(datas_train,k):
    n=datas_train.shape[1]
    centroids=np.zeros((k,n))
    for j in range(n):
        minJ=min(datas_train[:,j])
        maxJ=max(datas_train[:,j])
        centroids[:,j]=minJ+(maxJ-minJ)*np.random.rand(1,k)
    return centroids

def kMeans(datas_train, k, createCent = randCent):
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
                dist=abs(datas_train[i]-centroids[j])
                if dist<minDist:
                    minDist=dist
                    minIndex=j
            if clusterAssment[i][0]!=minIndex:
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2
        for cent in range(k):
            ptsInClust=datas_train[np.nonzero(clusterAssment[:,0] == cent)]
            if len(ptsInClust)!=0:
                centroids[cent,:]=np.mean(ptsInClust,axis=0)
    return centroids,clusterAssment

def getSSE(datas_train):
    loss=[]
    for i in range(0,5):
        print(i)
        distSum=np.sum(kMeans(datas_train,i+1)[1][:,1])
        loss.append(distSum)
    print(loss)
    plt.plot([x+1 for x in range(0,5)],loss)
    plt.show()

def save_result(datas_train,k,filename,shape,inverse):#将结果再次保存为图片
    centroids,clusterAssment=kMeans(datas_train,k)
    if not inverse:
        for i in range(len(centroids)):#黑白二值化
            if centroids[i]>0:
                centroids[i]=0
            else:
                centroids[i]=255
    else:
        for i in range(len(centroids)):
            if centroids[i]>0:
                centroids[i]=255
            else:
                centroids[i]=0

    new_img=clusterAssment[:,0]

    for i in range(len(new_img)):
        temp=new_img[i]
        new_img[i]=centroids[int(temp)]

    new_img=np.reshape(new_img,(shape[0],shape[1]))
    image=Image.fromarray(new_img)
    if image.mode == "F":
        image = image.convert('RGB')
    image.save('./Kmeans_result/{}.png'.format(filename.split('.')[0]))

k_dict={'black_kitten.jpg':2,'black_kitten_star.jpg':3,'black-white-kittens2.jpg':3,'cat_bed.jpg':3,'cat_grumpy.jpg':3,'cat_mouse.jpg':3,'cat-jumping-running-grass.jpg':3,'cutest-cat-ever-snoopy-sleeping.jpg':3,'grey-american-shorthair.jpg':3,'grey-cat-grass.jpg':2,'kitten9.jpg':3,'kitten16.jpg':2,'stripey-kitty.jpg':3,'the-black-white-kittens.jpg':3,'tortoiseshell_shell_cat.jpg':3,'young-calico-cat.jpg':3}
inverse_dict={'black_kitten.jpg':False,'black_kitten_star.jpg':False,'black-white-kittens2.jpg':False,'cat_bed.jpg':False,'cat_grumpy.jpg':True,'cat_mouse.jpg':False,'cat-jumping-running-grass.jpg':False,'cutest-cat-ever-snoopy-sleeping.jpg':True,'grey-american-shorthair.jpg':False,'grey-cat-grass.jpg':True,'kitten9.jpg':False,'kitten16.jpg':True,'stripey-kitty.jpg':False,'the-black-white-kittens.jpg':False,'tortoiseshell_shell_cat.jpg':True,'young-calico-cat':False}

def loop():
    for file in fileList:
        datas_train,shape=get_image(file)
        save_result(datas_train,k_dict[file],file,shape,inverse_dict[file])

loop()

