import tensorflow as tf

# 定义两个常数矩阵
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                       [2]])
# 矩阵相乘
product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    result = sess.run(product)
    print(result)
