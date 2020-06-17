#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/4/10 0010 下午 18:01
# @Author : West Field
# @File : logistic_regression.py

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def logistic_regression():
    '''
    逻辑回归：
    线性回归预测的是一个连续值，逻辑回归输出的是一个二分类的概率
    sigmoid函数是一个概率分布函数，给定某个输入，它将输出为一个概率值
    损失函数：平方差所惩罚的是与损失为同一个数量级的情形，对于分类问题最好使用交叉熵损失函数会更有效，交叉熵会输出一个更大的损失
    交叉熵损失函数经常用于分类问题中，特别是在神经网络做分类问题时，也经常使用交叉熵作为损失函数，此外，由于交叉熵涉及到计算每个类别的概率，所以交叉熵几乎每次都和sigmoid(或softmax)函数一起出现。
    :return:
    '''
#     读取数据
    data = pd.read_csv('./credit-a.csv', header=None)
    print(data.head())
    print(data.iloc[:, -1].value_counts())
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1].replace(-1, 0) # 这里改为负标签为0，正标签为1，-1、1适合于SVM
    print(type(x), type(y)) # pandas.core.frame.DataFrame  pandas.core.series.Series
#     建立模型
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4, input_shape=(15, ), activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
#     逻辑回归最后要使用sigmoid进行激活，输出一个概率值
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    print(model.summary())
#     编译模型，使用交叉熵损失函数，它在运行过程中，每运行完一个epoch它会计算metrics中的指标
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#     训练模型
    history = model.fit(x, y, epochs=100)
#     history.history是一个字典类型，记录了每个epoch时loss、acc的变化
    print(history.history.keys())
#     将loss、acc的变化绘图
    plt.plot(history.epoch, history.history.get('loss'))
    plt.show() # show()两次，loss、acc就出现在两张图中
    plt.plot(history.epoch, history.history.get('acc'))
    plt.show()

def softmax_regression():
    '''
    softmax分类：
    逻辑回归解决的是二分类问题，对于多分类的问题，使用softmax回归，它是逻辑回归的推广
    神经网络的原始输出不是一个概率值，实质上只是输入的数值做了复杂的加权和与非线性处理之后的一个值而已，那么可以使用softmax将这个输出变为概率分布。
    :return:
    '''
#     加载内置数据: Fashion MNIST数据集是服装图片数据，包含70000张灰度图像，涵盖10个类别
    (train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
    print(train_image.shape, train_label.shape, test_image.shape, test_label.shape)
    print(type(train_image),type(train_label)) # numpy.ndarray  numpy.ndarray
    plt.imshow(train_image[0])
    plt.show()
    print(train_image[0]) # 0-255之间的RGB值
#     数据归一化
    train_image = train_image/255
    test_image = test_image/255
#     建立模型
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # 使输入数据变成一个一维的向量
    model.add(tf.keras.layers.Dense(128, activation='relu')) # 神经元太少会丢弃掉一些信息，太多会过拟合
    model.add(tf.keras.layers.Dense(10, activation='softmax')) # 长度为10的概率输出
#     编译模型，softmax的损失函数有两个：categorical_crossentorpy、sparse_categorical_crossentropy，当label使用顺序数字编码的时候[0,1...,9]，使用的损失函数是sparse_categorical_crossentropy
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
#     训练模型
    model.fit(train_image, train_label, epochs=5)
#     模型评价
    model.evaluate(test_image, test_label)

# #     使用独热编码方式
#     train_label_onehot = tf.keras.utils.to_categorical(train_label)
#     test_label_onehot = tf.keras.utils.to_categorical(test_label)
#     print(train_label, train_label_onehot, test_label, test_label_onehot)
# #     建立模型
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # 使输入数据变成一个一维的向量
#     model.add(tf.keras.layers.Dense(128, activation='relu')) # 神经元太少会丢弃掉一些信息，太多会过拟合
#     model.add(tf.keras.layers.Dense(10, activation='softmax')) # 长度为10的概率输出
# #     编译模型，损失函数使用categorical_crossentropy
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# #     训练模型
#     model.fit(train_image, train_label_onehot, epochs=5)
# #     模型评价
#     model.evaluate(test_image, test_label_onehot)

if __name__ == '__main__':
    # 逻辑回归模型
    # logistic_regression()
    # softmax分类
    softmax_regression()
