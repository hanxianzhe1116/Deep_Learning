'''
    Variable变量
'''
import tensorflow as tf

state = tf.Variable(0, name='counter')  # 变量state={name:counter, value:0}
print(state.name)
one = tf.constant(1)  # 常量 1

new_value = tf.add(state, one)
update = tf.assign(state, new_value)  # 将new_value分配给state
# 若定义了Variable，则需要初始化
init = tf.initialize_all_variables()  # must have if define variable

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
