'''
    placeholder传值
'''
import tensorflow as tf

input1 = tf.placeholder(tf.float32)  # 若加入结构，输入两行两列,tf.placeholder(tf.float32,[2, 2])
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # feed_dict就是用来赋值的，格式为字典型。
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
