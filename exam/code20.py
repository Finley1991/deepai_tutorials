import tensorflow as tf

print(tf.__version__)
#图当中是看不到真实数据的，只有数据的形状和类型
x = tf.placeholder(dtype=tf.float32,shape=[None,784])
y = tf.placeholder(dtype=tf.float32,shape=[None,10])
w= tf.Variable(tf.random_normal(shape=[784,10],stddev=0.01,dtype=tf.float32))
b = tf.Variable(tf.zeros(shape=[10],dtype=tf.float32))

print(x)
print(w)

#  初始化图中全局变量的操作.
init = tf.global_variables_initializer()
#会话当中可以看到真实数据
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(w))


