import sys
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf  # 导入tensorflow V1
tf.disable_eager_execution()

f = open('./SVM_test_output.txt', 'w')
sys.stdout = f

datas_iris = load_iris()

x_vals = np.array([[x[0], x[3]] for x in datas_iris.data])  # 为了能使分类器可视化，选择二维数据
y_vals = np.array([1 if y == 0 else -1 for y in datas_iris.target])  # 二分类
print(x_vals.shape,y_vals.shape)

# 分离数据集与训练集
train_indices = np.random.choice(
    len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

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

# 开始训练，20000轮，每次喂入的批量为100
batch_size = 100

for i in range(20000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

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

f.close()
