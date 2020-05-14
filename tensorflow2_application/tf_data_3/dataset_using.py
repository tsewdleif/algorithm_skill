#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/4/10 0010 下午 18:08
# @Author : West Field
# @File : dataset_using.py

import tensorflow as tf
import numpy as np

def create_dataset():
    '''
    创建TensorSliceDataset类型的变量，其中每个元素都是一个tensor对象，这些对象被称为组件
    :return:
    '''
#   使用一维列表创建dataset
    dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
    for ele in dataset:
        print(ele)
        print(ele.numpy()) # tensor对象传换成numpy对象
#     使用二维列表创建dataset，其中的元素维度要一致
    dataset = tf.data.Dataset.from_tensor_slices([[1,2], [3, 4], [5, 6]])
    for ele in dataset:
        print(ele)
        print(ele.numpy()) # tensor对象传换成numpy对象
#     使用字典来创建dataset
    dataset = tf.data.Dataset.from_tensor_slices({'a': [1, 2, 3, 4], 'b': [6,7,8,9], 'c':[12,13,14,15]})
    for ele in dataset:
        print(ele)
#     使用ndarray创建dataset
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4, 5, 6]))
    for ele in dataset:
        print(ele)
        print(ele.numpy()) # tensor对象传换成numpy对象
#     dataset的使用，取前四个元素
    for ele in dataset.take(4):
        print(ele)
        print(ele.numpy())

def transform_data():
    '''
    dataset对数据的变换
    :return:
    '''
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4, 5, 6, 7]))
#     对数据进行乱序，参数为需要乱序的数据的的个数
    dataset = dataset.shuffle(6)
    for ele in dataset:
        print(ele.numpy())
#     对数据进行重复，每一次都是乱序的，参数为重复的次数，如果不写默认无限次重复
    dataset = dataset.repeat(count=3)
    for ele in dataset:
        print(ele.numpy())
#     取出batch_size大小的数据
    dataset = dataset.batch(3)
    for ele in dataset:
        print(ele.numpy())
#     使用函数对数据进行变换，根据函数对每一个元素变换
    dataset = dataset.map(tf.square)
    for ele in dataset:
        print(ele.numpy())

def input_dataset():
    '''
    dataset作为输入，训练模型
    mnist是手写数据集
    :return:
    '''
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
#     数据归一化
    train_images = train_images/255
    test_images = test_images/255
#     创建train_images的dataset
    ds_train_images = tf.data.Dataset.from_tensor_slices(train_images)
    print(train_images.shape, ds_train_images)
#     建立label的dataset
    ds_train_lab = tf.data.Dataset.from_tensor_slices(train_labels)
#     将images和label合并在一起
    ds_train = tf.data.Dataset.zip((ds_train_images, ds_train_lab))
    print(ds_train)
#     数据洗牌变换，设置batch
    ds_train = ds_train.shuffle(10000).repeat().batch(64)
#     创建模型
    model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                 tf.keras.layers.Dense(128, activation='relu'),
                                 tf.keras.layers.Dense(10, activation='softmax')])
#     编译模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     训练模型，在ds_train上进行训练，它既包含训练的图片也包含训练的label，设置每个epoch训练的步数
    steps_per_epoch = train_images.shape[0]//64
    model.fit(ds_train, epochs=5, steps_per_epoch=steps_per_epoch)

def add_validation_data():
    '''
    使用dataset训练模型的过程中添加dataset的validation_data，在model.fit的时候添加一个测试数据进去，就可以看到在每一步的训练过程中model在验证集上的结果
    :return:
    '''
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    #     数据归一化
    train_images = train_images / 255
    test_images = test_images / 255
    #     创建train_images的dataset
    ds_train_images = tf.data.Dataset.from_tensor_slices(train_images)
    print(train_images.shape, ds_train_images)
    #     建立label的dataset
    ds_train_lab = tf.data.Dataset.from_tensor_slices(train_labels)
    #     将images和label合并在一起
    ds_train = tf.data.Dataset.zip((ds_train_images, ds_train_lab))
    print(ds_train)
    #     数据洗牌变换，无限repeat数据，设置batch
    ds_train = ds_train.shuffle(10000).repeat().batch(64)
    # 创建验证集的dataset
    ds_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    # 对于验证集数据无需shuffle操作，默认repeat数据，设置batch大小
    ds_test = ds_test.batch(64)
    #     创建模型
    model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                 tf.keras.layers.Dense(128, activation='relu'),
                                 tf.keras.layers.Dense(10, activation='softmax')])
    #     编译模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #     训练模型，在ds_train上进行训练，它既包含训练的图片也包含训练的label，设置验证集，设置每个epoch训练的步数和每个epoch验证的步数
    steps_per_epoch = train_images.shape[0] // 64
    model.fit(ds_train, epochs=5, steps_per_epoch=steps_per_epoch, validation_data=ds_test, validation_steps=10000//64) # 10000是test数据集的样本个数


if __name__ == '__main__':
    # 创建dataset
    # create_dataset()
#     对数据进行变换
#     transform_data()
#   使用dataset作为输入训练模型
#     input_dataset()
#   添加验证集的dataset的模型训练
    add_validation_data()
