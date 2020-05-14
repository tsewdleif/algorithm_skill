#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/4/12 0012 下午 23:33
# @Author : West Field
# @File : cat_dog_category.py

import tensorflow as tf
import numpy as np
import glob

def cat_dog_category():
    '''
    猫狗识别
    :return:
    '''
#     使用tf.data读取数据
    image_filenames = glob.glob('./dc/train/*.jpg') # 获取所有训练集图片路径
    image_filenames = np.random.permutation(image_filenames) # 乱序处理
    labels = list(map(lambda x: float(x.split('\\')[1].split('.')[0] == 'cat'), image_filenames)) # 若==成立，表示当前标签为cat，label=1； 若当前标签为dog，则label=0。
    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, labels)) # 传换成dataset格式
    print(dataset)
#     读取图像内容
    dataset = dataset.map(pre_read)
    dataset = dataset.shuffle(300)
    dataset = dataset.repeat() # 无参数时默认一直重复下去，在fit()里边用epochs参数控制重复结束
    dataset = dataset.batch(32)
    print(dataset)
#     创建模型，一般卷积好的模型是3层Conv2D+MaxPooling2D
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(200,200,1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    print(model.summary())
#     编译模型
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#     训练模型
    model.fit(dataset, epochs=10, steps_per_epoch=int(25000/32))


def pre_read(img_filename, label):
    '''
    处理图片
    :param img_filename:
    :param lable:
    :return:
    '''
    image = tf.io.read_file(img_filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.rgb_to_grayscale(image) # 传换成单通道灰度图像，对于图像识别，是否是彩色图像不影响识别，所以转换成单通道，使得运算速度加快
    image = tf.image.resize(image, (200, 200)) # resize成200*200尺寸
    image = tf.reshape(image, [200, 200, 1]) # reshape成200*200*1单通道灰度图像
    image = image / 255 #tf.image.per_image_standardization(image) # 图像标准化
    label = tf.reshape(label, [1])
    return image, label


if __name__ == '__main__':
    # 猫狗识别
    cat_dog_category()
