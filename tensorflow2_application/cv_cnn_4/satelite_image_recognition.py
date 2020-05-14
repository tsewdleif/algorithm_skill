#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/4/12 0012 上午 11:38
# @Author : West Field
# @File : satelite_image_recognition.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# pathlib是面向对象的路径工具，比os.path更好用一些
import pathlib
import random

def data_extration():
    '''
    用tf.data来提取硬盘上的数据，作为卷积神经网络的输入
    数据包括飞机、湖两类航拍图片
    :return:
    '''
    data_dir = './dataset/2_class'
    # 得到一个路径对象，它有很多方法，能帮助提取里边的路径
    data_root = pathlib.Path(data_dir)
#     对里边的目录迭代
    for item in data_root.iterdir():
        print(item)
#    提取所有路径下的图片路径，函数参数为正则表达式，即所有路径下的所有图片
    all_image_path = list(data_root.glob('*/*'))
    print(len(all_image_path))
    print(all_image_path[:3], all_image_path[-3:])
#     变成字符串格式的路径
    all_image_path = [str(path) for path in all_image_path]
    print(all_image_path[10:12])
#     乱序处理
    random.shuffle(all_image_path)
    print(all_image_path[10:12])
#     得到所有的目录名，对图片进行编码，一个为0一个为1
    label_names = sorted(item.name for item in data_root.glob('*/'))
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print(label_to_index)
#   获取所有图片对应的编码
    all_image_label = [label_to_index[pathlib.Path(p).parent.name] for p in all_image_path]
    print(all_image_label[:5], all_image_path[:5])
#     随机显示3张图片
    index_to_label = dict((v, k) for k, v in label_to_index.items())
    print(index_to_label)
    for n in range(3):
        image_index = random.choice(range(len(all_image_path)))
        plt.imshow(load_preprocess_image(all_image_path[image_index]))
        plt.show()
        # 输出对应图片的标签名
        print(index_to_label[all_image_label[image_index]])
        print()
#     建立图片的dataset
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_path) # 建立图片路径的dataset
    image_dataset = path_ds.map(load_preprocess_image)
    for img in image_dataset.take(1):
        print(img)
    # 建立图片label的dataset
    label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
    for label in label_dataset.take(10):
        print(label.numpy())
#     zip成一个dataset，它的shape就是一个元组，第一部分是一个图片，第二部分是一个对应的label
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
#     划分为训练集和验证集，20%为验证集
    train_count = int(0.8 * len(all_image_path))
    test_count = len(all_image_path) - train_count
    train_dataset = dataset.skip(test_count)
    test_datset = dataset.take(test_count)
#     对训练集做变换，shuffle保证每次迭代的时候都是乱序输入到神经网络中。这里没有设置repeat，不写就默认无限制的重复产生数据。
    BATCH_SIZE = 16
    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(BATCH_SIZE)
#     对测试集无需shuffle，只需设置batch
    test_datset = test_datset.batch(BATCH_SIZE)
    # 使用tf.data构建dataset数据管道速度非常快，在第一次读取数据之后就会将缓存加载到硬盘上，速度不差于tfrecord
    return train_dataset, test_datset, train_count, test_count, BATCH_SIZE


def load_preprocess_image(image_path):
    '''
    加载单张图片内容
    :param image_path: 图片路径
    :return:
    '''
    #   用tensorflow读取图片内容
    img_raw = tf.io.read_file(image_path)
    print(img_raw)
    #     解码图片，decode_image()是通用的解析图片的方法，它不会返回图片的shape，所以使用decode_jgep()
    img_tensor = tf.image.decode_jpeg(img_raw, channels=3) # channels=3表示是彩色图像
    # 让图像resize到需要的大小
    img_tensor = tf.image.resize(img_tensor, [256, 256])
    print(img_tensor.shape, img_tensor.dtype, img_tensor)
    #     输入神经网络的数据最好进行标准化
    img_tensor = tf.cast(img_tensor, tf.float32)  # 转换数据类型
    img_tensor = img_tensor / 255
    # img_tensor = img_tensor.numpy()  # 将tensor转换成ndarray
    # print(img_tensor.max(), img_tensor.min())
    return img_tensor


def image_recognition():
    '''
    对卫星图片进行识别
    :return:
    '''
    # 获取训练集、测试集、训练集样本数目、测试集样本数目、batch_size
    train_data, test_data, train_count, test_count, BATCH_SIZE = data_extration()
    # 在构建好数据的输入管道之后，下面来构造这个模型
    # 建立顺序模型，卷积核个数递增
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling2D()) # 变成了(None, 1024)的数据格式，None是样本个数
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
#     输出层，因为这里是二分类，所以输出层为1个神经元（也可以为2个，差不多，不过一个会少很多训练参数）。使用sigmoid激活函数（也可以使用softmax激活函数，在二分类时两者等价）。
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    print(model.summary())
#     编译模型，二分类的损失函数是binary_crossentropy
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#     步数
    steps_per_epoch = train_count/BATCH_SIZE
    validation_steps = test_count/BATCH_SIZE
#     训练模型，告诉模型多少个step为一个epoch结束
    history = model.fit(train_data, epochs=30, steps_per_epoch=steps_per_epoch, validation_data=test_data, validation_steps=validation_steps)
#     画图
    print(history.history.key())
    plt.plot(history.epoch, history.history.get('acc'), label='acc')
    plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
    plt.legend()
    plt.show()
    plt.plot(history.epoch, history.history.get('loss'), label='loss')
    plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
    plt.legend()
    plt.show()

def add_batch_normalization():
    '''
    添加批归一化层
    批标准化通常在卷积层或密集连接层之后使用，批标准化是分批次进行的，所以叫做批标准化。
    在模型进行预测的时候，使用的均值和方差是整个训练集的均值和方差，即在训练时记录每个batch的均值和方差，训练完之后求整个训练集的均值和方差。
    原论文讲，在CNN中批标准化用于激活函数之前，但实际上用于激活函数之后效果可能更好。
    :return:
    '''
    '''
    传统机器学习中，标准化也叫归一化，指将数据映射到指定范围。使模型看到的样本之间更加相似，使泛化性能更好。
    数据标准化两种形式：标准化((x-均值)/标准差)和归一化((x-min)/(max-min))。
    批标准化是将分散数据统一的一种做法，也是优化神经网络的一种方法。
    批标准化不仅在输入模型之前对数据做标准化，在网络每次变换之后都应考虑对数据标准化。
    批标准化解决的是梯度消失与梯度爆炸问题。好处是：具有正则化的效果、提高模型泛化能力、加速收敛。
    对于特别深的网络，只有包含多个BatchNormalization层才能进行训练。
    '''
    # 获取训练集、测试集、训练集样本数目、测试集样本数目、batch_size
    train_data, test_data, train_count, test_count, BATCH_SIZE = data_extration()
    # 在构建好数据的输入管道之后，下面来构造这个模型
    # 建立顺序模型，卷积核个数递增
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3)))
    # 卷积层后添加批标准化
    model.add(tf.keras.layers.BatchNormalization())
    # 添加激活层，也可以直接将激活层放到上边的卷积层中，就变成了先激活后批归一化
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.MaxPooling2D()) # 卷积层后无需添加批标准化
    model.add(tf.keras.layers.Conv2D(128, (3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Conv2D(128, (3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(256, (3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Conv2D(256, (3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(512, (3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(512, (3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(1024, (3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.GlobalAveragePooling2D())  # 变成了(None, 1024)的数据格式，None是样本个数
    model.add(tf.keras.layers.Dense(1024))
    # 全连接层添加批标准化
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    #     输出层
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    print(model.summary())
    #     编译模型
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    #     步数
    steps_per_epoch = train_count / BATCH_SIZE
    validation_steps = test_count / BATCH_SIZE
    #     训练模型，告诉模型多少个step为一个epoch结束
    history = model.fit(train_data, epochs=30, steps_per_epoch=steps_per_epoch, validation_data=test_data,
                        validation_steps=validation_steps)
    #     画图
    print(history.history.key())
    plt.plot(history.epoch, history.history.get('acc'), label='acc')
    plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
    plt.legend()
    plt.show()
    plt.plot(history.epoch, history.history.get('loss'), label='loss')
    plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 提取数据
    # data_extration()
    # 图像识别
    # image_recognition()
    #  添加批归一化batch_normalization层
    add_batch_normalization()
