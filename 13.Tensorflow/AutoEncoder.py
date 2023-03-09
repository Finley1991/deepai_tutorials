import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

class Net:

    def __init__(self):
        #输入X，要求输出也是X
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,784])
        # self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        #定义dtopout占位符
        self.dropout = tf.placeholder(dtype=tf.float32)

        self.w1 = tf.Variable(tf.truncated_normal([784,1000],stddev=0.1),dtype=tf.float32)
        self.b1 = tf.Variable(tf.zeros([1000]),dtype=tf.float32)

        self.w2 = tf.Variable(tf.truncated_normal([1000, 500], stddev=0.1), dtype=tf.float32)
        self.b2 = tf.Variable(tf.zeros([500]), dtype=tf.float32)

        self.w3 = tf.Variable(tf.truncated_normal([500, 784],stddev=0.1), dtype=tf.float32)
        self.b3 = tf.Variable(tf.zeros([784]), dtype=tf.float32)


    def forward(self):
        y1 = tf.nn.relu(tf.matmul(self.x,self.w1) + self.b1)
        #对中间层输出神经元dropout限制
        y1=tf.nn.dropout(y1,keep_prob=self.dropout)

        y1 = tf.nn.relu(tf.matmul(y1, self.w2) + self.b2)
        # 对中间层输出神经元dropout限制
        y1 = tf.nn.dropout(y1, keep_prob=self.dropout)

        self.output = tf.matmul(y1,self.w3) + self.b3

    def backward(self):
        self.optimizer = tf.train.AdamOptimizer().minimize(self.error)

    def loss(self):
        # self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output,labels=self.x))\
        #              +0.0001*tf.reduce_sum(self.w1**2)+0.0001*tf.reduce_sum(self.w2**2)
        # self.error = -tf.reduce_mean(self.y*tf.log(self.output))
        self.error = tf.reduce_mean((self.output - self.x)**2)

    def validate(self):
        y = tf.equal(tf.argmax(self.output,1),tf.argmax(self.x,1))
        self.accuracy = tf.reduce_sum(tf.cast(y,dtype=tf.float32))

if __name__ == '__main__':
    net = Net()
    net.forward()
    net.loss()
    net.backward()
    net.validate()
    net.init = tf.global_variables_initializer()

    mnist = input_data.read_data_sets("../tensorflow_test/dataset/mnist/", one_hot=True)

    with tf.Session() as sess:

        sess.run(net.init)

        for i in range(50000):
            xs,_ = mnist.train.next_batch(100)
            _accuracy,_error,_ = sess.run([net.accuracy,net.error,net.optimizer],feed_dict={net.x:xs,net.dropout:0.9})
            if i%100 == 0:
                print(_error)
                print(_accuracy)
                xs, _ = mnist.test.next_batch(1)
                _output = sess.run(net.output, feed_dict={net.x: xs, net.dropout: 1.0})
                im = np.reshape(_output,[28,28])
                plt.imshow(im,cmap="gray")
                plt.pause(0.001)

