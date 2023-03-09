import tensorflow as tf
import numpy as np
import PIL.Image as pimg
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../tensorflow_test/dataset/mnist', one_hot=True)

batch_size = 100


class Net:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.dropout = tf.placeholder(dtype=tf.float32)

        self.w1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 16], stddev=tf.sqrt(1 / 16), dtype=tf.float32))
        self.b1 = tf.Variable(tf.zeros(shape=[16], dtype=tf.float32))  # 14*14*16

        self.w2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 32], stddev=tf.sqrt(1 / 32), dtype=tf.float32))
        self.b2 = tf.Variable(tf.zeros(shape=[32], dtype=tf.float32))  # 7*7*32

        self.w_fcn1 = tf.Variable(tf.truncated_normal(shape=[7 * 7 * 32, 7 * 7 * 64], stddev=tf.sqrt(1 / (7 * 7 * 64)), dtype=tf.float32))
        self.b_fcn1 = tf.Variable(tf.zeros(shape=[7 * 7 * 64], dtype=tf.float32))  # [7*7*64]

        # 开始反卷积
        self.w3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], stddev=tf.sqrt(1 / 32), dtype=tf.float32))
        self.b3 = tf.Variable(tf.zeros(shape=[32], dtype=tf.float32))  # 14*14*32

        self.w4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 8, 32], stddev=tf.sqrt(1 / 8), dtype=tf.float32))
        self.b4 = tf.Variable(tf.zeros(shape=[8], dtype=tf.float32))  # 28*28*8

        self.wo = tf.Variable(tf.truncated_normal(shape=[28 * 28 * 8, 784], stddev=tf.sqrt(1 / 784), dtype=tf.float32))
        self.bo = tf.Variable(tf.zeros(shape=[784], dtype=tf.float32))  # 784

    def forward(self, batch_size):
        self.y0 = tf.reshape(self.x, shape=[-1, 28, 28, 1])

        self.y1 = tf.nn.relu(tf.nn.conv2d(input=self.y0, filter=self.w1, strides=[1, 1, 1, 1], padding='SAME') + self.b1)
        self.y11 = tf.nn.max_pool(value=self.y1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self._y1 = tf.nn.dropout(self.y11, keep_prob=self.dropout)

        self.y2 = tf.nn.relu(tf.nn.conv2d(input=self._y1, filter=self.w2, strides=[1, 1, 1, 1], padding='SAME') + self.b2)
        self.y22 = tf.nn.max_pool(value=self.y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self._y2 = tf.nn.dropout(self.y22, keep_prob=self.dropout)

        self.y = tf.reshape(self._y2, [-1, 7 * 7 * 32])

        self.y_fcn1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(self.y, self.w_fcn1) + self.b_fcn1))

        self.y_conv = tf.reshape(self.y_fcn1, [-1, 7, 7, 64])

        # 反卷积
        self.y3 = tf.nn.relu(tf.nn.conv2d_transpose(self.y_conv, self.w3, [batch_size, 14, 14, 32], [1, 2, 2, 1], 'SAME') + self.b3)
        self._y3 = tf.nn.dropout(self.y3, keep_prob=self.dropout)


        self.y4 = tf.nn.relu(tf.nn.conv2d_transpose(self._y3, self.w4, [batch_size, 28, 28, 8], [1, 2, 2, 1], 'SAME') + self.b4)
        self._y4 = tf.nn.dropout(self.y4, keep_prob=self.dropout)

        self.y00 = tf.reshape(self._y4, [-1, 28 * 28 * 8])
        self.ys = tf.matmul(self.y00, self.wo) + self.bo
        self.y_out = tf.nn.sigmoid(self.ys)

        # self.y_out = self.ys

    def loss(self):
        # self.error = tf.reduce_mean((self.x - self.y_out) ** 2)
        self.error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=self.ys))

    def backward(self):
        self.optimizer = tf.train.AdamOptimizer().minimize(self.error)

if __name__ == '__main__':

    net = Net()
    net.forward(batch_size)
    net.loss()
    net.backward()
    net.init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(net.init)

        a = []
        b = []

        for i in range(5000):
            xs, _ = mnist.train.next_batch(batch_size)
            _error, _ = sess.run(fetches=[net.error, net.optimizer], feed_dict={net.x: xs, net.dropout: 0.9})
            if i % 2 == 0:
                xss, _ = mnist.validation.next_batch(batch_size)
                output = sess.run(fetches=net.y_out, feed_dict={net.x: xss, net.dropout: 1.0})
                outimg = np.reshape(output[0], [28, 28])
                img = pimg.fromarray(np.uint8(outimg*255))
                print(i)
                print('train _error:', _error)

                # a.append(i)
                # b.append(_error)
                # plt.clf()
                # plt.plot(a,b)
                # plt.show()
                plt.imshow(outimg,cmap="gray")
                # plt.imshow(img,cmap="gray")
                plt.pause(0.01)
