import tensorflow as tf
sess = tf.compat.v1.Session()
a = tf.constant(10)
b = tf.constant(12)
res = sess.run(a+b)
print(res)