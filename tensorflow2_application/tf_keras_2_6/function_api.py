#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/4/10 0010 下午 18:04
# @Author : West Field
# @File : function_api.py

import tensorflow as tf
import matplotlib.pyplot as plt

def function_api():
    '''
    函数式API：
    每一层都是一个类，它的实例是可以当作函数一样调用。
    函数式API好处：可以建立多输入多输出的模型
    :return:
    '''
    #     加载内置数据: Fashion MNIST数据集是服装图片数据，包含70000张灰度图像，涵盖10个类别
    (train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
    print(train_image.shape, train_label.shape, test_image.shape, test_label.shape)
    print(train_image[0])  # 0-255之间的RGB值
    #     数据归一化
    train_image = train_image / 255
    test_image = test_image / 255
    #     建立模型
    input = tf.keras.Input(shape=(28, 28))
    x = tf.keras.layers.Flatten()(input)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=input, outputs=output)
    print(model.summary())
    #     编译模型，softmax的损失函数有两个：categorical_crossentorpy、sparse_categorical_crossentropy，当label使用顺序数字编码的时候[0,1...,9]，使用的损失函数是sparse_categorical_crossentropy
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    #     训练模型
    history = model.fit(train_image, train_label, epochs=30, validation_data=(test_image, test_label))
#     绘图查看loss在训练集和验证集上的变化
    plt.plot(history.epoch, history.history.get('loss'), label='loss')
    plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
    plt.legend()
    plt.show()
    #     绘图查看acc在训练集和验证集上的变化
    plt.plot(history.epoch, history.history.get('acc'), label='acc')
    plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #     函数式API
    function_api()
