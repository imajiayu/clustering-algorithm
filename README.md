# 计算机视觉 1950509 马家昱 作业1

## K-Means聚类

### 算法原理

对于给定的样本集，按照样本之间的距离大小，将样本集划分为K个簇。让簇内的点尽量紧密的连在一起，而让簇间的距离尽量的大。



算法步骤：

1. 选择初始化的 k 个样本作为初始聚类中心 ![[公式]](https://www.zhihu.com/equation?tex=a%3D%7Ba_1%2Ca_2%2C%E2%80%A6a_k%7D) ；

2. 针对数据集中每个样本 ![[公式]](https://www.zhihu.com/equation?tex=x_i) 计算它到 k 个聚类中心的距离并将其分到距离最小的聚类中心所对应的类中；

3. 针对每个类别 ![[公式]](https://www.zhihu.com/equation?tex=a_j) ，重新计算它的聚类中心 ![[公式]](https://www.zhihu.com/equation?tex=a_j%3D%5Cfrac%7B1%7D%7B%5Cleft%7C+c_i+%5Cright%7C%7D%5Csum_%7Bx%5Cin+c_i%7Dx) （即属于该类的所有样本的质心）；

4. 重复上面 2 3 两步操作，直到达到某个中止条件（迭代次数、最小误差变化等）。

   

K值选择：SSE手肘法

![img](https://pic3.zhimg.com/80/v2-5ca4a5fe0b06b25a2b97262abb401a16_720w.jpg)



当 K < 3 时，曲线急速下降；当 K > 3 时，曲线趋于平稳，通过手肘法我们认为拐点 3 为 K 的最佳值。

### 代码实现

使用sklearn.datasets中的鸢尾花数据集进行算法实现，使用二维数据进行输入以便于图形化

```python
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
plt.show
```

标准的参考分类结果如图所示：

![](https://github.com/imajiayu/clustering-algorithm/blob/main/screenshots/1.png)  

下面为K-Means聚类算法需要用的函数：

```python
#计算欧氏距离
def distEclud(vecA,vecB):
    return math.sqrt(sum((vecA-vecB)**2))

#取得k个中心，其质心随机
datas_train=np.array([[x[2],x[3]] for x in datas])
def randCent(datas_train,k):
    n=datas_train.shape[1]
    centroids=np.zeros((k,n))#质心为k*n的矩阵
    for j in range(n):#随机产生质心
        minJ=min(datas_train[:,j])
        maxJ=max(datas_train[:,j])
        centroids[:,j]=minJ+(maxJ-minJ)*random.rand(1,k)
    return centroids
```

K-Means算法的主循环

```python
#K-means 聚类算法
def kMeans(datas_train, k, distMeans =distEclud, createCent = randCent):
    m=datas_train.shape[0]
    clusterAssment=np.zeros((m,2))
    centroids=createCent(datas_train,k)
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        for i in range(m):#每一个样本点
            minDist=math.inf
            minIndex=-1
            for j in range(k):#每一个质心
                dist=distMeans(datas_train[i],centroids[j])
                if dist<minDist:
                    minDist=dist
                    minIndex=j
            if clusterAssment[i][0]!=minIndex:#若有样本点的质心改变，说明正在收敛
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2
        for cent in range(k):#更改每个中心的质心
            ptsInClust=datas_train[np.nonzero(clusterAssment[:,0] == cent)]
            centroids[cent,:]=np.mean(ptsInClust,axis=0)
    return centroids,clusterAssment
```

使用SSE手肘法寻找最适合的K值：

```python
def getSSE():
    loss=[]
    for i in range(0,10):#测试从1到10的k值
        distSum=np.sum(kMeans(datas_train,i+1)[1][:,1])
        loss.append(distSum)
    print(loss)
    plt.plot([x+1 for x in range(0,10)],loss)
    plt.show()
```

结果如下：

![3](3.png)

当k=3时，画出实际分类的结果

```python
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
```

![2](2.png)

### 实际应用

将原始图片转化为灰度图，并对像素矩阵进行标准化。

```python
def get_image(filename):#获得标准化后的灰度图片矩阵
    pli_im=Image.open('./data/imgs/{}'.format(filename))
    temp=pli_im.convert('L')
    matrix=np.asarray(temp) 

    matrix=normalize_image(matrix)

    datas_train=matrix.reshape(matrix.shape[0]*matrix.shape[1],1)

    return datas_train,matrix.shape
```

```python
def save_result(datas_train,k,filename,shape,inverse):#将结果再次保存为图片
    centroids,clusterAssment=kMeans(datas_train,k)
    if not inverse:
        for i in range(len(centroids)):#二值化
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
```

效果如下：

![black-white-kittens2](data/imgs/black-white-kittens2.jpg)![black-white-kittens2](Kmeans_result/black-white-kittens2.png)

将所有图片与GroundTruth进行对比，计算正确率：

```python
def compare(gt_matrix,re_matrix):#计算两个矩阵中不同元素的个数与总像素数的比值
    assert gt_matrix.shape==re_matrix.shape
    total=gt_matrix.shape[0]*gt_matrix.shape[1]
    return 1-len(np.argwhere(gt_matrix!=re_matrix))/total

for re_file in result_fileList:
    for gt_file in gt_fileList:
        if re_file==gt_file:
            pli_gt=Image.open('./data/gt/{}'.format(gt_file))
            gt_matrix=np.asarray(pli_gt.convert('L'))
            pli_re=Image.open('./Kmeans_result/{}'.format(re_file))
            re_matrix=np.asarray(pli_re.convert('L'))
            print(re_file.split('.')[0]+' '+str(compare(gt_matrix,re_matrix)))
```

结果如下：

![image-20210504234551588](../../../AppData/Roaming/Typora/typora-user-images/image-20210504234551588.png)

## HAC聚类

### 算法原理

通过计算不同类别数据点间的相似度来创建一棵有层次的嵌套聚类树。在聚类树中，不同类别的原始数据点是树的最低层，树的顶层是一个聚类的根节点。

![image-20210505014734107](../../../AppData/Roaming/Typora/typora-user-images/image-20210505014734107.png)

算法步骤：

1. 将所有样本都看作各自一类
2. 定义类间距离计算公式
3. 选择距离最小的一堆元素合并成一个新的类
4. 重新计算各类之间的距离并重复上面的步骤
5. 直到所有的原始元素划分成指定数量的类

### 代码实现

使用sklearn.datasets中的鸢尾花数据集进行算法实现，使用二维数据进行输入以便于图形化

类间距离定义为average

```python
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
```

聚类结果如下：

![4](4.png)

### 实际应用

HAC算法的时间复杂度过高，因此先将原始图片压缩后进行聚类，同比例压缩GroundTruth进行比较。

```python
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
```

```python
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
```

效果如下：

<img src="data/imgs/grey-american-shorthair.jpg" alt="grey-american-shorthair" style="zoom:50%;" /><img src="HAC_result/grey-american-shorthair.png" alt="grey-american-shorthair"  /> 

将所有图片与GroundTruth进行对比，计算正确率：

```python
def compare(gt_matrix,re_matrix):#计算两个矩阵中不同元素的个数与总像素数的比值
    assert gt_matrix.shape==re_matrix.shape
    total=gt_matrix.shape[0]*gt_matrix.shape[1]
    return 1-len(np.argwhere(gt_matrix!=re_matrix))/total

for re_file in result_fileList:
    for gt_file in gt_fileList:
        if re_file==gt_file:
            pli_gt=Image.open('./data/gt/{}'.format(gt_file))
            gt_matrix=np.asarray(pli_gt.convert('L'))
            pli_re=Image.open('./HAC_result/{}'.format(re_file))
            re_matrix=np.asarray(pli_re.convert('L'))
            print(re_file.split('.')[0]+' '+str(compare(gt_matrix,re_matrix)))
```

结果如下：

![image-20210505132608019](../../../AppData/Roaming/Typora/typora-user-images/image-20210505132608019.png)

## k-最邻近(k-NN)分类器

### 原理

k-最邻近算法是一种基本分类和回归方法。只需在与训练数据集中寻找与输入实例最邻近的K个实例，则该实例属于K个实例中的多数。如图所示，K选择的不同影响了其分类器的准确度。
![avatar](./1.png)

### 实现过程

#### 使用python实现分类器

1. 数据集使用sklearn.iris,其data为一个(750,4)向量，分别表示萼片长度、宽度、花瓣长度、宽度，target共有三类

```py
datas_iris=load_iris()#导入数据集
x_data=np.array(datas_iris['data'][0:150])
y_data=np.array(datas_iris['target'][0:150])

datas=np.column_stack((x_data,y_data))#将数据集变为维度为(150,5)的矩阵，第五列表示类别

datas_train=datas[0:150:2]#划分训练集和测试集
datas_test=datas[0:150:2]
```

2. 计算距离，这里使用的是欧式距离

```py
def get_distances(x0,datas_train):#计算一个数据与整个训练集的距离，并由小到大排序
    distances=[]
    for x_t in datas_train:
        distant=sqrt(np.sum((x0[0:4]-x_t[0:4])**2))#欧氏距离
        distances.append([distant,x_t[4]])
    return sorted(distances)
```

3. 排序，选择最近的K个样本，取其中最多的种类为结果

```py
def predict(x0,datas_train,k):#根据k值预测x0的类别
    distances=get_distances(x0,datas_train)
    predict_target_list=[distances[i][1] for i in range(0,k)]
    votes=Counter(predict_target_list)#统计前k个数据中每个类别的数量
    return votes.most_common(1)[0][0]#取最多的那个
```

4. 遍历测试集，获得该K值下的准确度

```py
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
```

5. 循环迭代K值，求得最佳的k

```py
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
```

结果如下：

![avatar](./2.png)

#### 将分类器应用至图片分类中

使用数据集为sklearn.digits,数字识别，每一张图片由8x8的矩阵构成，data为一个(1797,64)向量，target为0-9共10个数字。数字"0"的样本如下：
![avatar](./3.png)

以其中1500个样本为训练集，迭代求最佳k值
完整代码：

```py
from sklearn.datasets import load_digits
import numpy as np
from math import sqrt
from collections import Counter
import matplotlib.pyplot as plt
import sys

f = open('./KNN_application_output.txt', 'w')
sys.stdout = f

datas_digits = load_digits()  # 导入数据集
x_data = np.array(datas_digits['data'][0:1797])
y_data = np.array(datas_digits['target'][0:1797])

datas = np.column_stack((x_data, y_data))  # 将数据集变为维度为(1797,65)的矩阵，第65列表示类别

datas_train = datas[0:1500]  # 划分训练集和测试集
datas_test = datas[1500:1797]


def get_distances(x0, datas_train):  # 计算一个数据与整个训练集的距离，并由小到大排序
    distances = []
    for x_t in datas_train:
        distant = sqrt(np.sum((x0[0:64]-x_t[0:64])**2))  # 欧氏距离
        distances.append([distant, x_t[64]])
    return sorted(distances)


def predict(x0, datas_train, k):  # 根据k值预测x0的类别
    distances = get_distances(x0, datas_train)
    predict_target_list = [distances[i][1] for i in range(0, k)]
    votes = Counter(predict_target_list)  # 统计前k个数据中每个类别的数量
    return votes.most_common(1)[0][0]  # 取最多的那个


def get_precision(datas_train, datas_test, k):  # 遍历测试集，计算准确度
    result = [0, 0]
    for x0 in datas_test:
        pre = predict(x0, datas_train, k)
        print("预测值为", x0[64], "真实值为", pre, x0[64] == pre)
        if x0[64] == pre:
            result[0] = result[0]+1
        else:
            result[1] = result[1]+1
    return result[0]/(result[0]+result[1])


def iteration(datas_train, datas_test):  # 迭代求使得精确度最高的k值
    best_precision = 0
    parameter = ()
    for k in range(1, 10):
        precision = get_precision(datas_train, datas_test, k)
        print("k=", k, "准确度=", precision)
        if precision > best_precision:
            best_precision = precision
            parameter = (k, best_precision)
    return parameter


print("最佳k与其准确率为", iteration(datas_test, datas_train))

f.close()
```

结果如下：

![avatar](./4.png)

## 支持向量机(SVM)分类器

### 原理

支持向量机是一个解决二分类问题的分类器。若要找到一个超平面将两组数据分开，可以采用多种分类方法。但为了使分类器的泛化能力达到最大，我们需要使离超平面最近的点（支持向量）与超平面的距离最远。

![avatar](./5.png)

### 实现过程

数据集仍然使用sklearn.iris，使用tensorflow训练模型，训练集与数据集的比例为4：1。为了使分类器可视化，只选取四维数据中的二维，以便于画图。

1. 载入数据集

```py
datas_iris = load_iris()

x_vals = np.array([[x[0], x[3]] for x in datas_iris.data])  # 为了能使分类器可视化，选择二维数据
y_vals = np.array([1 if y == 0 else -1 for y in datas_iris.target])  # 二分类

# 分离数据集与训练集
train_indices = np.random.choice(
    len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]
```

2. 定义模型

```py
# 喂入数据
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 要训练的两个参数
A = tf.Variable(tf.random_normal(shape=[2, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# 定义线性模型
model_output = tf.subtract(tf.matmul(x_data, A), b)

# L2范数，防止过拟合
l2_norm = tf.reduce_sum(tf.square(A))

# 定义损失函数为Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
alpha = tf.constant([0.01])
classification_term = tf.reduce_mean(tf.maximum(
    0., tf.subtract(1., tf.multiply(model_output, y_target))))
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# 初始化session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
```

3. 开始训练

```py
# 开始训练，20000轮，每次喂入的批量为100
batch_size = 100

for i in range(20000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
```

4. 可视化训练集

```py
# 可视化部分
[[a1], [a2]] = sess.run(A)
[[b]] = sess.run(b)
slope = -a2/a1  # 斜率
y_intercept = b/a1
# 分界线上的点
best_fit = []
x1_vals = [d[1] for d in x_vals]

for i in x1_vals:
    best_fit.append(slope*i+y_intercept)

setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == 1]
setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == 1]
not_setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == -1]
not_setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == -1]

plt.plot(setosa_x, setosa_y, 'o', label='I. setosa')
plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=3)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()
```

1. 计算测试集的准确度

```py
true_answer = 0
false_answer = 0

a = np.array([a2, a1]).T
for i in range(len(x_vals_test)):
    result = 1 if np.dot(x_vals_test[i][::-1], a)-b > 0 else -1
    print("预测值为:", result, "真实值为:", y_vals_test[i], result == y_vals_test[i])
    if result == y_vals_test[i]:
        true_answer += 1
    else:
        false_answer += 1

print("准确率为：", true_answer/(true_answer+false_answer))
```

结果如下：

![avatar](./6.png)
![avatar](./7.png)

### 将分类器应用到图像分类中

依然使用sklearn.digits数字识别数据集。由于SVM只能解决二分类问题，这里使用**一类对余类**方法，即对每一个类都将其设置为1，其余类设置为-1，需要训练多组模型，达到多分类效果。

* CPU:Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz   2.59 GHz
* GPU：NVIDIA GTX1660ti 不使用cuda工具包的情况下速度较慢

完整代码如下：

```py
import sys
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf  # 导入tensorflow V1
tf.disable_eager_execution()

f = open('./SVM_application_output.txt', 'w')
sys.stdout = f

datas_digits = load_digits()
x_data=np.array(datas_digits['data'])
y_data=np.array(datas_digits['target'])
datas=np.column_stack((x_data,y_data))

# 分离数据集与训练集
train_indices = np.random.choice(
    len(x_data), round(len(x_data)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_data))) - set(train_indices)))

datas_train=datas[train_indices]
datas_test=datas[test_indices]

x_vals_test=np.array([x[0:64] for x in datas_test])
y_vals_test=np.array([x[64] for x in datas_test])

number_0_vals=[x for x in datas_train if x[64]==0]
number_1_vals=[x for x in datas_train if x[64]==1]
number_2_vals=[x for x in datas_train if x[64]==2]
number_3_vals=[x for x in datas_train if x[64]==3]
number_4_vals=[x for x in datas_train if x[64]==4]
number_5_vals=[x for x in datas_train if x[64]==5]
number_6_vals=[x for x in datas_train if x[64]==6]
number_7_vals=[x for x in datas_train if x[64]==7]
number_8_vals=[x for x in datas_train if x[64]==8]
number_9_vals=[x for x in datas_train if x[64]==9]

x0_vals=np.array([x[0:64] for x in number_0_vals])
y0_vals=np.array([x[64] for x in number_0_vals])

x1_vals=np.array([x[0:64] for x in number_1_vals])
y1_vals=np.array([x[64] for x in number_1_vals])

x2_vals=np.array([x[0:64] for x in number_2_vals])
y2_vals=np.array([x[64] for x in number_2_vals])

x3_vals=np.array([x[0:64] for x in number_3_vals])
y3_vals=np.array([x[64] for x in number_3_vals])

x4_vals=np.array([x[0:64] for x in number_4_vals])
y4_vals=np.array([x[64] for x in number_4_vals])

x5_vals=np.array([x[0:64] for x in number_5_vals])
y5_vals=np.array([x[64] for x in number_5_vals])

x6_vals=np.array([x[0:64] for x in number_6_vals])
y6_vals=np.array([x[64] for x in number_6_vals])

x7_vals=np.array([x[0:64] for x in number_7_vals])
y7_vals=np.array([x[64] for x in number_7_vals])

x8_vals=np.array([x[0:64] for x in number_8_vals])
y8_vals=np.array([x[64] for x in number_8_vals])

x9_vals=np.array([x[0:64] for x in number_9_vals])
y9_vals=np.array([x[64] for x in number_9_vals])

x_vals_list=[x0_vals,x1_vals,x2_vals,x3_vals,x4_vals,x5_vals,x6_vals,x7_vals,x8_vals,x9_vals]
y_vals_list=[y0_vals,y1_vals,y2_vals,y3_vals,y4_vals,y5_vals,y6_vals,y7_vals,y8_vals,y9_vals]

x_data = tf.placeholder(shape=[None, 64], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 要训练的两个参数
A = tf.Variable(tf.random_normal(shape=[64, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# 定义线性模型
model_output = tf.subtract(tf.matmul(x_data, A), b)

# L2范数，防止过拟合
l2_norm = tf.reduce_sum(tf.square(A))

# 定义损失函数为Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
alpha = tf.constant([0.01])
classification_term = tf.reduce_mean(tf.maximum(
    0., tf.subtract(1., tf.multiply(model_output, y_target))))
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# 初始化session
init = tf.global_variables_initializer()

# 开始训练，20000轮，每次喂入的批量为100
batch_size = 100
model_parameter=[]
def train():#训练十次

    def run(x_vals,y_vals):
        for i in range(20000):
            rand_index = np.random.choice(len(x_vals), size=batch_size)
            rand_x = x_vals[rand_index]
            rand_y = np.transpose([y_vals[rand_index]])
            sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    for i in range(0,10):
        print(i)
        sess = tf.Session()
        sess.run(init)
        run(x_vals_list[i],y_vals_list[i])
        a_list = sess.run(A)
        B = sess.run(b)
        model_parameter.append([[x[0] for x in a_list],B[0][0]])

train()

true_answer = 0
false_answer = 0

def predict(x_vals,parameter_list):#在每一个模型下进行预测，取最可能的值
    max=0
    loc=-1
    for i in range(0,10):
        result=np.dot(x_vals[::-1],np.array(parameter_list[i][0][::-1]).T)-parameter_list[i][1]
        if result>0 and result>max:
            max=result
            loc=i
    return i

for i in range(len(x_vals_test)):
    result = predict(x_vals_test[i],model_parameter)
    print("预测值为:", result, "真实值为:", y_vals_test[i], result == y_vals_test[i])
    if result == y_vals_test[i]:
        true_answer += 1
    else:
        false_answer += 1

print("准确率为：", true_answer/(true_answer+false_answer))

f.close()

```

结果如下：
![avatar](./8.png)

## 两者的差异与权衡

1. k-NN天然支持多分类问题，单一SVM只能解决二分类问题
2. k-NN为数据依赖型分类器，不需要训练模型，只需要数据集即可，SVM需要训练模型中的参数再进行预测