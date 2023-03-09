import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist= input_data.read_data_sets("./dataset/mnist",one_hot=True)
import matplotlib.pyplot as plt

class Net:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,784])
        self.y = tf.placeholder(dtype=tf.float32,shape=[None,10])
        self.w= tf.Variable(tf.random_normal(shape=[784,10],stddev=0.01,dtype=tf.float32))
        self.b = tf.Variable(tf.zeros(shape=[10],dtype=tf.float32))
    def forward(self):
        self.output = tf.matmul(self.x,self.w)+self.b
    def backward(self):
        self.error = tf.reduce_mean(tf.square(self.y - self.output))
        self.optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(self.error)
    def accuracy(self):
        result = tf.equal(tf.argmax(self.y,axis=1),tf.argmax(self.output,axis=1))
        self.acc = tf.reduce_mean(tf.cast(result,tf.float32))

if __name__ == "__main__":
    net = Net()
    net.forward()
    net.backward()
    net.accuracy()

    init = tf.global_variables_initializer()

    a=[]
    b=[]
    c=[]

    with tf.Session() as sess:
        sess.run(init)
        for i in range(50000):
            xs,ys = mnist.train.next_batch(100)#批次
            # print(xs.shape)
            # print(ys.shape)
            # print(net.x.shape)
            # print(net.y.shape)
            # exit()
            error, accuracy, _ = sess.run(fetches=[net.error, net.acc, net.optimizer], feed_dict={net.x: xs, net.y: ys})
            if i%100 == 0:
                xss, yss = mnist.validation.next_batch(100)
                _error,out,_accuracy = sess.run(fetches=[net.error,net.output,net.acc], feed_dict={net.x: xss, net.y: yss})
                label=np.argmax(yss[0])
                output=np.argmax(out[0])
                print("train_error:",error,"validate_error:",_error)
                print("label:",label,"output:",output)
                print("epoch:",i)
                print("train_accuracy:", accuracy)
                print("validate_accuracy:", _accuracy)
                array = np.reshape(xss[0],[28,28])*255
                # print(array)
                a.append(i)
                b.append(error)
                c.append(_error)
                plt.clf()
                train, = plt.plot(a,b,linewidth=1,color="red")
                validate, = plt.plot(a,c,linewidth=1,color="blue")
                plt.legend([train,validate],["train","validate"],loc="best",fontsize=10)
                plt.title("loss drop chart",color="red",fontsize=15)
                plt.pause(0.1)
    plt.ioff()