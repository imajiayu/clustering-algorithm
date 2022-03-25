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
