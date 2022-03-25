import matplotlib.pyplot as plt
import numpy as np
import math
import os
from PIL import Image

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
    datas_train=datas_train.tolist()
    for i in range(len(datas_train)):
        datas_train[i].append(i)
    return datas_train,matrix.shape

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
        print(len(classes))
        i,j=merge_arg(classes)
        classes[i]+=classes[j]
        classes.pop(j)
    return classes

def save_result(datas_train,filename,shape,class_num,inverse):#将结果再次保存为图片
    classes=HAC(datas_train,class_num)

    new=[]
    for i in range(class_num):
        average=sum([x[0] for x in classes[i]])/len(classes[i])
        temp=np.array(classes[i])
        if average>0:
            temp[:,0]=0 if not inverse else 255
        else:
            temp[:,0]=255 if not inverse else 0

        new+=temp.tolist()
    
    new.sort(key=lambda x:x[1])
    new_img=np.array([x[0] for x in new])
    
    new_img=np.reshape(new_img,(shape[0],shape[1]))
    image=Image.fromarray(new_img)

    if image.mode == "F":
        image = image.convert('RGB')
    image.save('./HAC_result/{}.png'.format(filename.split('.')[0]))

def loop():
    for file in fileList:
        datas_train,shape=get_image(file)
        save_result(datas_train,file,shape,2,False)
        
loop()