#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/4/10 0010 下午 18:02
# @Author : West Field
# @File : overfitting.py

import tensorflow as tf
import matplotlib.pyplot as plt

'''
在具体实践中，可通过查看损失函数值随时间的变化曲线，来判断学习速率的选取是合适的。
合适的学习速率，损失函数随时间下降，直到一个底部。不合适的学习速率，损失函数可能会发生震荡。
优化器：(optimizer) 是编译模型的所需的两个参数之一。你可以先实例化一个优化器对象，然后将它传入model.compile或者你可以通过名称来调用优化器。在后一种情况下，将使用优化器的默认参数。
SGD：随机梯度下降优化器，SGD 和 min batch 是同一个意思，抽取m个小批量（独立同分布）样本，通过计算他们平梯度均值。
RMSprop：经验上被证明有效且实用的深度学习网络优化算法，增加了一个衰减系数来控制历史信息的获取多少，RMSProp会对学习率进行衰减。
Adam算法可以看做是修正后的Momentum+RMSProp算法，通常被认为对超参数的选择相当鲁棒，学习率建议为 0.001。
'''
'''
网络容量：
可以认为与网络中的可训练参数成正比。网络中的神经单元数越多，层数越多，神经网络的拟合能力越强。但是训练速度、难度越大，越容易产生过拟合。
如何提高网络的拟合能力：一种显然的想法是增大网络容量，增加层或增加隐藏神经元个数。
这两种方法哪种更好呢：单纯的增加神经元个数对于网络性能的提高并不明显，增加层会大大提高网络的拟合能力，这也是为什么现在深度学习的层越来越深的原因。
单层的神经元个数，不能太小，太小的话，会造成信息瓶颈，使得模型欠拟合。
'''

def dropout_avoid_overfitting():
    '''
    dropout抑制过拟合与超参数选择
    :return:
    '''
    #     加载内置数据: Fashion MNIST数据集是服装图片数据，包含70000张灰度图像，涵盖10个类别
    (train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
    print(train_image.shape, train_label.shape, test_image.shape, test_label.shape)
    print(train_image[0])  # 0-255之间的RGB值
    #     数据归一化
    train_image = train_image / 255
    test_image = test_image / 255
#     使用独热编码方式
    train_label_onehot = tf.keras.utils.to_categorical(train_label)
    test_label_onehot = tf.keras.utils.to_categorical(test_label)
    print(train_label, train_label_onehot, test_label, test_label_onehot)
#     建立模型
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # 使输入数据变成一个一维的向量
    model.add(tf.keras.layers.Dense(128, activation='relu')) # 神经元太少会丢弃掉一些信息，太多会过拟合
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax')) # 长度为10的概率输出
#     编译模型，损失函数使用categorical_crossentropy
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['acc'])
#     训练模型，使用validation_data来验证每个epoch之后的metrics指标的变化
    history = model.fit(train_image, train_label_onehot, epochs=10, validation_data=(test_image, test_label_onehot))
    print(history.history.keys())
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
    '''
    由上边的指标图可以发现模型过拟合，那么可以通过dropout机制实现抑制过拟合。
    dropout原理类似于集成算法，在训练的时候每个epoch随机丢弃掉隐藏层的某些神经元，即训练完之后每个epoch形成了一个模型，而在预测的时候不使用dropout，即所有训练时学习到的神经元参数都用上。
    为什么说Dropout可以解决过拟合？
    (1)取平均的作用。先回到标准的模型即没有dropout，我们用相同的训练数据去训练 5 个不同的神经网络，一般会得到 5 个不同的结果，此时我们可以采用 5 个结果取均值”或者“多数取胜的投票策略”去决定最终结果。
    (2)减少神经元之间复杂的共适应关系。因为dropout程序导致两个神经元不一定每次都在一个 dropout 网络中出现。这样权值的更新不再依赖于有固定关系的隐含节点的共同作用，阻止了某些特征仅仅在其它特定特征下才有效果的情况。
    (3)Dropout 类似于性别在生物进化中的角色。物种为了生存往往会倾向于适应这种环境，环境突变则会导致物种难以做出及时反应，性别的出现可以繁衍出适应新环境的变种，有效的阻止过拟合，即避免环境改变时物种可能面临的灭绝。 
    '''
    '''
    超参数选择的原则：
    理想的模型是刚好在欠拟合和过拟合的界线上，也就是正好拟合数据。
    首先开发一个过拟合的模型：添加更多的层、让每一层变得更大、训练更多的轮次。
    然后，抑制过拟合：dropout、正则化、图像增强(数据量不够大时使用)。
    再次，调节超参数：学习速率、隐藏层单元数、训练轮次。
    构建网络的总原则：
    一、增大网络容量，直到过拟合
    二、采取措施抑制过拟合
    三、继续增大网络容量，直到过拟合
    '''
    #     建立模型
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    # 添加dropout
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))  # 长度为10的概率输出
    #     编译模型，损失函数使用categorical_crossentropy
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',
                  metrics=['acc'])
    #     训练模型，使用validation_data来验证每个epoch之后的metrics指标的变化
    history = model.fit(train_image, train_label_onehot, epochs=10, validation_data=(test_image, test_label_onehot))
    print(history.history.keys())
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

def small_network_avoid_overfitting():
    '''
    使用简单的网络结构抑制过拟合：
    使用更小的网络迫使学习到更关键的特征，避免过拟合
    :return:
    '''
    #     加载内置数据: Fashion MNIST数据集是服装图片数据，包含70000张灰度图像，涵盖10个类别
    (train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
    print(train_image.shape, train_label.shape, test_image.shape, test_label.shape)
    print(train_image[0])  # 0-255之间的RGB值
    #     数据归一化
    train_image = train_image / 255
    test_image = test_image / 255
    #     使用独热编码方式
    train_label_onehot = tf.keras.utils.to_categorical(train_label)
    test_label_onehot = tf.keras.utils.to_categorical(test_label)
    print(train_label, train_label_onehot, test_label, test_label_onehot)
    #     建立模型，减小网络规模
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))  # 长度为10的概率输出
    #     编译模型，损失函数使用categorical_crossentropy
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',
                  metrics=['acc'])
    #     训练模型，使用validation_data来验证每个epoch之后的metrics指标的变化
    history = model.fit(train_image, train_label_onehot, epochs=10, validation_data=(test_image, test_label_onehot))
    print(history.history.keys())
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
    # dropout抑制过拟合与超参数选择
    dropout_avoid_overfitting()
    # 小网络结构抑制过拟合
    small_network_avoid_overfitting()
