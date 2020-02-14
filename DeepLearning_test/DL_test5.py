'''
    建造神经网络，及数据变化可视化
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 建造一个神经网络
# 构建数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 代表创建300行的-1,1之间的值，后面加了一个维度，300*1
noise = np.random.normal(0, 0.05, x_data.shape)  # 添加一个随机噪点
y_data = np.square(x_data) - 0.5 + noise

# 定义输入输出的格式
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# 定义隐藏层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)  # 定义：输入一个参数，隐藏层输出10个参数（10个神经元）
prediction = add_layer(l1, 10, 1, activation_function=None)  # 相当于输出层，接收的数据为l1隐藏层的数据，
# 均方差loss求和
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))  # reduction_indices表示处理的维度

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    fig = plt.figure()  # 生成图片框
    ax = fig.add_subplot(1, 1, 1)  # 连续性的图
    ax.scatter(x_data, y_data)  # 先将数据用点绘制
    plt.ion()  # 连续起来
    plt.show()
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        # 每50步输出一次loss值
        if i % 50 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            # 输出loss值
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)  # 设置一条红色的线，表示变化
            plt.pause(0.5)  # 暂停时间
    plt.pause(0)