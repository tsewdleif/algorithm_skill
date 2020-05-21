#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/4/12 0012 下午 20:02
# @Author : West Field
# @File : imdb_category.py

import tensorflow as tf
import matplotlib.pyplot as plt

def movie_review_category():
    '''
    tf.keras处理序列问题：电影评论分类（正面和负面评论）
    :return:
    '''
#     电影评论数据获取
    data = tf.keras.datasets.imdb
#     加载电影评论数据，可以看到索引有到两万多的，这里可以选择只要前一万个词（指的是字典中对应的前一万个高频词）的索引，一万个词以外的不给它编码抛弃掉
    (x_train, y_train), (x_test, y_test) = data.load_data(num_words=10000)
    print(x_train.shape, y_train.shape, x_train[0]) #数据中每个元素是一个单词对应的索引
#     对文本中的整数值进行编码。文本编码方式：tf-idf编码、word2vec embedding密集向量编码、one-hot编码。
#     最好的文本编码方法是，把文本训练成密集向量。keras把它封装成了一个层。
    print([len(x) for x in x_train]) #可以看到评论文本长度不相同，需要把评论填充或截断到固定长度，以方便放到神经网络里边训练。
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, 300) # 设置长度为300
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, 300)
    print([len(x) for x in x_train])
#     建立模型
    model = tf.keras.Sequential()
#     训练成密集向量，同word2vec，参数10000是输入数据的最大维度即最多10000个单词，50表示要映射成的向量的维度，300表示输入序列数据长度
    model.add(tf.keras.layers.Embedding(10000, 50, input_length=300))
#     经过上边一层以后，数据从(25000, 300)变成了(25000, 300, 50)的张量。添加flatten层变成一维向量，以便输入后边的全连接层。
    model.add(tf.keras.layers.Flatten())
#     添加全连接层
    model.add(tf.keras.layers.Dense(128, activation='relu'))
#     输出层
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    print(model.summary())
#     编译模型，metrics是监控指标
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
#     训练模型
    history = model.fit(x_train, y_train, epochs=15, batch_size=256, validation_data=(x_test, y_test))
#     绘图
    print(history.history.keys())
    plt.plot(history.epoch, history.history['acc'], 'r')
    plt.plot(history.epoch, history.history['val_acc'], 'b--')
    plt.show()
    plt.plot(history.epoch, history.history['loss'], 'r')
    plt.plot(history.epoch, history.history['val_loss'], 'b--')
    plt.show()

def solve_overfitting():
    '''
    解决过拟合问题：1、添加dropout层，2、L2,L1正则化
    :return:
    '''
    #     电影评论数据获取
    data = tf.keras.datasets.imdb
    #     加载电影评论数据
    (x_train, y_train), (x_test, y_test) = data.load_data(num_words=10000)
    print(x_train.shape, y_train.shape, x_train[0])  # 数据中每个元素是一个单词对应的索引
    print(type(x_train),type(y_train))
    print(y_train)
    #     设置长度为300
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, 300)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, 300)
    #     建立模型
    model = tf.keras.Sequential()
    #     训练成密集向量
    model.add(tf.keras.layers.Embedding(10000, 50, input_length=300))
    #     经过上边一层以后，数据从(25000, 300)变成了(25000, 300, 50)的张量。添加flatten层变成一维向量，以便输入后边的全连接层。
    # model.add(tf.keras.layers.Flatten())
    # 使用全局平均池化代替flatten展平，由(300, 50)变成(50)，相比flatten的(15000)大大的减少了参数，可以抑制过拟合
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    #     添加全连接层，添加L2正则化
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # 添加dropout层
    model.add(tf.keras.layers.Dropout(0.5))
    #     输出层
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    print(model.summary())
    #     编译模型，metrics是监控指标
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
    #     训练模型
    history = model.fit(x_train, y_train, epochs=15, batch_size=256, validation_data=(x_test, y_test))
    #     绘图
    print(history.history.keys())
    plt.plot(history.epoch, history.history['acc'], 'r')
    plt.plot(history.epoch, history.history['val_acc'], 'b--')
    plt.show()
    plt.plot(history.epoch, history.history['loss'], 'r')
    plt.plot(history.epoch, history.history['val_loss'], 'b--')
    plt.show()


if __name__ == '__main__':
    # 电影评论数据分类
    # movie_review_category()
#     解决过拟合问题
    solve_overfitting()
