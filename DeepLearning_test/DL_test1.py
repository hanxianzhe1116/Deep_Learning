import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3  # 0.1相当于weights权重，0.3是biases偏置项

#  ## create tensorflow structure start ##  #
# Variable是weights的参数变量，用随机数列生成，[1]代表一维，取值范围-1，1
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# biases初始为零
biases = tf.Variable(tf.zeros([1]))
y = Weights * x_data + biases
# 损失函数采用均方差
loss = tf.reduce_mean(tf.square(y - y_data))
# optimizer优化器，减少误差提升准确度
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 梯度下降，学习效率，小于1
train = optimizer.minimize(loss)  # 最小化
init = tf.initialize_all_variables()  # 初始化
# ##create tensorflow structure end ## #

# 初始化的两种方法
# sess = tf.Session()
# sess.run(init)  # very important step
with tf.Session() as sess:
    sess.run(init)
    # 开始训练，200步
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            # 训练的每步权重和偏置
            print(step, sess.run(Weights), sess.run(biases))
