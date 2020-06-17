#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/4/12 0012 上午 9:15
# @Author : West Field
# @File : fashion_mnist_recognition.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

def cnn_image_recognition():
    '''
    使用卷积神经网络进行图像识别，可以和多层感知器神经网络相比，它的特点和优势
    训练网络学习的是卷积核，先对卷积核按照某种规则随机初始化，然后进行网络训练学习
    对于图像任务，如果用多层感知器网络，参数会非常多，如果用卷积神经网络，只需学习卷积核即可，大大的减少了网络的参数
    卷积层用于提取图像的特征，不同的卷积核提取的特征不同，如用于提取边缘轮廓信息的卷积核
    CNN使得图像逐渐变得小而厚，小是池化层的作用，变小使得视野变大，厚是卷积层的作用使得提取的特征变多（卷积核个数等于输出通道数，每个通道对应一个特征）
    :return:
    '''
    # 查看GPU是否可用
    print(tf.test.is_gpu_available())
    # 加载数据集
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
    print(train_images[0:2])
#     前边使用多层感知器网络对数据的变换是扁平化数据，这里使用CNN对图像的处理是扩张成四维数据
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
    print(train_images.shape, test_images.shape) # 这里是(None, 28, 28, 1)表示是黑白图像，即(None, height, width, channel)
    print(train_images[0:2])
#     构建模型，一般第一层是卷积层，因为卷积层对图像特征的提取能力远大于全连接层
    model = tf.keras.Sequential() # 顺序模型
    # 卷积层，32是feature(特征，即输出的通道的个数)的个数，即卷积核个数，即通道个数，卷积核一般选择3*3、5*5，一般用relu进行激活，padding方式采用same即保持卷积后的图像尺寸不变，默认padding是valid方式
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=train_images.shape[1:], activation='relu', padding='same'))
#     查看当前model层的输出数据的形状
    print(model.output_shape)
#     池化层，一般默认是2*2池化，池化层使得图像的视野不断地变大
    model.add(tf.keras.layers.MaxPool2D())
#     再添加卷积层，feature个数以2的指数递增，这样拟合能力更强，这里设置为64个卷积核
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
#     在链接全连接层之前，卷积后是4维的数据，不能直接链接，这里需要变成一维数据，两种做法，添加flatten层做扁平化或者做全局池化，flatten会引入很多的参数，所以用全局池化比较好
#     池化层，GlobalAveragePooling2D和MaxPool2D不同，GlobalAveragePooling2D代表全局的池化，即每个通道取一个平均数
    model.add(tf.keras.layers.GlobalAveragePooling2D())
#     全连接层
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    print(model.summary())
#     编译模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
#     训练模型
    history = model.fit(train_images, train_labels, epochs=30, validation_data=(test_images, test_labels))
    print(history.history.keys())
#     acc训练过程绘图
    plt.plot(history.epoch, history.history.get('acc'), label='acc')
    plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
    plt.legend()
    plt.show()
    #     loss训练过程绘图
    plt.plot(history.epoch, history.history.get('loss'), label='loss')
    plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
    plt.legend()
    plt.show()
#   由图可以看出，模型过拟合，而且train数据准确率没有达到100%，说明这个网络的拟合能力有待提高
#   优化：增加网络深度或隐藏层单元个数以增强拟合能力，添加dropout层抑制过拟合

def cnn_optimize():
    '''
    对上边的CNN进行优化：增加网络层、增加每层神经元个数，添加dropout层
    :return:
    '''
    # 查看GPU是否可用
    print(tf.test.is_gpu_available())
    # 加载数据集
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
    # 扩张成四维数据
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
    print(train_images.shape, test_images.shape)
    # 构建模型
    model = tf.keras.Sequential()  # 顺序模型
    # 卷积层，第一次隐藏层单元数过少，可能会产生信息的瓶颈，即可能会漏掉一些信息
    model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=train_images.shape[1:], activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), padding='same')
    # 池化层
    model.add(tf.keras.layers.MaxPool2D())
    # dropout层
    model.add(tf.keras.layers.Dropout(0.5))
    # 再添加卷积层
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'), padding='same')
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'), padding='same')
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Dropout(0.5))
    # 再添加卷积层
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'), padding='same')
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'), padding='same')
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Dropout(0.5))
    # 再添加卷积层
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'), padding='same')
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'), padding='same')
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dropout(0.5))
    # 全连接层
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    print(model.summary())
    # 编译模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    # 训练模型
    history = model.fit(train_images, train_labels, epochs=30, validation_data=(test_images, test_labels))
    print(history.history.keys())
    # acc训练过程绘图
    plt.plot(history.epoch, history.history.get('acc'), label='acc')
    plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
    plt.legend()
    plt.show()
    # loss训练过程绘图
    plt.plot(history.epoch, history.history.get('loss'), label='loss')
    plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 使用卷积神经网络进行图像识别
    # cnn_image_recognition()
#     改进的CNN进行图像识别
    cnn_optimize()
