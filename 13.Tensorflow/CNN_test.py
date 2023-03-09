import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

class Net:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,784])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.w1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 16],stddev=0.01))#k_h,k_w,c_in,c_out
        self.b1 = tf.Variable(tf.zeros([16]))#14
        self.w2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 32],stddev=0.01))
        self.b2 = tf.Variable(tf.zeros([32]))#7
        self.w3 = tf.Variable(tf.truncated_normal(shape=[7*7*32,512],stddev=0.01))
        self.b3 = tf.Variable(tf.zeros([512]))
        self.w4 = tf.Variable(tf.truncated_normal(shape=[512, 10],stddev=0.01))
        self.b4 = tf.Variable(tf.zeros([10]))

    def forward(self):
        '''
        在tensorflow中，stride的一般形式是[1，x，y，1]
        第一个1表示：在batch维度上的滑动步长为1，即不跳过任何一个样本
        x表示：卷积核的水平滑动步长
        y表示：卷积核的垂直滑动步长
        最后一个1表示：在通道维度上的滑动步长为1，即不跳过任何一个颜色通道
        '''
        y = tf.reshape(self.x,[-1,28,28,1])
        y = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(y,self.w1,[1,1,1,1],padding="SAME")+self.b1,training=True))
        y = tf.nn.max_pool(y,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")#NHWC
        y = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(y,self.w2,[1,1,1,1],padding="SAME")+self.b2,training=True))
        y = tf.nn.max_pool(y, [1, 2, 2, 1], [1, 2, 2, 1],padding="SAME")

        y = tf.reshape(y, [-1,7*7*32])
        y = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(y,self.w3)+self.b3,training=True))
        self.ys = tf.matmul(y, self.w4) + self.b4
        self.output = tf.nn.softmax(self.ys)

    def backward(self):
        self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.ys, labels=self.y))#MCE loss
        # self.error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.ys, labels=self.y))#BCE loss
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.error)

    def validate(self):
        y = tf.equal(tf.argmax(self.output,1),tf.argmax(self.y,1))
        self.accuracy = tf.reduce_sum(tf.cast(y,dtype=tf.float32))

if __name__ == '__main__':

    ckpt_path = r"./param.ckpt"

    net = Net()
    net.forward()
    net.backward()
    net.validate()
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    mnist = input_data.read_data_sets("dataset/mnist/", one_hot=True)

    a=[]
    b=[]
    c=[]
    with tf.Session() as sess:
        sess.run(init)
        #恢复参数
        saver.restore(sess,ckpt_path)

        for i in range(5000):
            xs,ys = mnist.train.next_batch(100)
            error,_ = sess.run(fetches=[net.error,net.optimizer],feed_dict={net.x:xs,net.y:ys})

            if i%10 == 0:
                xss, yss = mnist.validation.next_batch(100)
                _output, _error, _accuracy= sess.run(fetches=[net.output, net.error, net.accuracy],feed_dict={net.x: xss, net.y: yss})
                print(i)
                print(np.argmax(yss[0]))
                print(np.argmax(_output[0]))
                print("error:", _error,'\t' "accuracy:",_accuracy)
                #保存参数
                saver.save(sess, ckpt_path)
                a.append(i)
                b.append(error)
                c.append(_error)
                plt.xlabel("x")
                plt.ylabel("y")
                plt.clf()
                train_line, = plt.plot(a,b, linewidth=2.0, color='red')
                validate_line, = plt.plot(a,c, linewidth=2.0, color='blue')
                plt.legend([train_line, validate_line], ["train", "validation"], loc='upper right', fontsize=10)

                plt.title("Loss drop chart", fontsize=15, color='g')
                plt.pause(0.01)

    plt.ioff()

