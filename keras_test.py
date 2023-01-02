import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels) = mnist.load_data() #加载数据
print('shape of train images is ',train_images.shape)
print('shape of train labels is ',train_labels.shape)
print('train labels is ',train_labels)
print('shape of test images is ',test_images.shape)
print('shape of test labels is',test_labels.shape)
print('test labels is',test_labels)
from keras import models
from keras import layers
network = models.Sequential()
# layers.Dense()的第一个参数指定的是当前层的输出维度
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))#全连接层
network.add(layers.Dense(10,activation='softmax'))#softmax层，返回10个概率，每个概率表示表示当前图片属于数字的概率
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# 处理训练集
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')/255
# 处理测试集
test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')/255
# 处理标签
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
network.fit(train_images,train_labels,epochs=5,batch_size=128)
test_loss,test_acc = network.evaluate(test_images,test_labels)
print("test_loss:",test_loss)
print("test_acc:",test_acc)
