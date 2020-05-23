#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/4/9 0009 下午 17:24
# @Author : West Field
# @File : linear_regression.py

# 之前的tensorflow是1.11.0版本，现在安装tensorflow2.0，来学习2.0的使用。
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def single_variable():
    '''
    单变量线性回归算法：
    x代表学历，f(x)代表收入，f(x)=ax+b
    目标：预测函数与真实值之间的整体误差最小
    损失函数：均方差
    优化的目标：找到合适的参数a、b，使(f(x)-y)^2越小越好
    如何优化：使用梯度下降算法
    :return:
    '''
    # 读取数据
    data = pd.read_csv('./Income1.csv')
    # 绘制散点图
    # plt.scatter(data.Education, data.Income)
    # plt.show()
    x = data.Education
    y = data.Income
    print(type(x), type(y)) # pandas.core.series.Series  pandas.core.series.Series
    print(x)
    print(y)
#     定义顺序模型
    model = tf.keras.Sequential()
#     添加层，设置输入输出
    model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
#     查看model的整体形状
#     print(model.summary())
#     编译model，可以看成是配置model的过程，优化方法是梯度下降算法（深度学习里边，局部极值点不是问题，它会随机初始化很多初始值，总会找到全局最小值），损失函数是均方差
    model.compile(optimizer='adam', loss='mse') # 这里adam默认学习速率是0.01
#     开始训练model，并记录下它的history，训练过程是一个最小化损失函数的过程，它要训练很多次，即epochs，对所有数据训练的次数
    history = model.fit(x, y, epochs=5000)
    # print(history)
#     使用模型进行预测
    print(model.predict(x))
    print(model.predict(pd.Series([20])))

def multilayer_perceptron():
    '''
    多层感知器的实现：
    x是在不同平台投放的广告量，label是收益
    单层神经元无法拟合异或运算，神经元要求数据必须是线性可分的，这个问题使得神经网络的发展停滞了很多年。
    为了解决线性不可分的问题，采取在网络的输入端和输出端之间插入更多的神经元，即多层感知器模型。
    :return: 
    '''
#     读取数据
    data = pd.read_csv('./Advertising.csv')
    # print(data.head())
    plt.scatter(data.TV, data.sales) # 在TV上的广告投放量和收益之间的关系
    # plt.show()
    x = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]
    print(type(x), type(y)) # pandas.core.frame.DataFrame       pandas.core.series.Series
    print(x)
    print(y)
#     定义模型，直接写上层，省的再去添加，这里是两层，隐含层10个神经元，输出层一个神经元
    model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(3,), activation='relu'),
                           tf.keras.layers.Dense(1)])
    print(model.summary())
#     编译模型
    model.compile(optimizer='adam', loss='mse')
#     训练模型
    model.fit(x, y, epochs=100)
#     预测数据
    test = data.iloc[:10, 1:-1]
    print(model.predict(test))
    print(data.iloc[:10, -1])


if __name__ == "__main__":
    # 单变量的线性回归
    # single_variable()
    # 多层感知器模型
    multilayer_perceptron()
