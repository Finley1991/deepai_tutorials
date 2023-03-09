import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist
import numpy as np

#创建数据存放地址
DATA_URL = r'D:\pycharmprojects\pythonProject\tensorflow_demo\dataset\mnist.npz'
""
#下载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data(DATA_URL)
#获得整体数据地址
path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
#获得训练集图像、标签数据和测试集图像、标签数据
with np.load(path) as data:
  train_examples = data['x_train']
  train_labels = data['y_train']
  test_examples = data['x_test']
  test_labels = data['y_test']

#创建对应的训练集和测试集
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

#打乱训练集数据，并且获取每批次训练集和测试集数据的数量
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

#创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu',),
    tf.keras.layers.Dense(10, activation='softmax')
])
#创建优化器、损失、精度衡量
model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
#训练模型
model.fit(train_dataset, epochs=10)
#测试模型
model.evaluate(test_dataset)