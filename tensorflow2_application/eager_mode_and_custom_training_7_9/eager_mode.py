#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/4/13 0013 下午 13:05
# @Author : West Field
# @File : eager_mode.py

import tensorflow as tf
import numpy as np

def eager_mode():
    '''
    eager模式是和1.0中图模式相对的一种模式
    对于自定义循环和训练，tf.keras封装的有些过度，所以tensorflow2.0提供了eager模式和自定义的循环和训练
    这种高低搭配，可以帮助我们快速的搭建网络，也可以添加一些自定义的动作
    eager模式可以使我们可以立即评估操作产生的结果，而无需构建计算图。方便了调试模型，可以叫它tensorflow的交互模式。
    eager模式下可以使用python控制流而不是图控制流，简化了动态模型的创建，tensorflow会立即执行并将其值返回给python。
    eager模式也支持大多数tensorflow操作和GPU加速。tf.tensor对象引用具体值而不是计算图中节点的符号句柄。
    eager模式下tensorflow可以与numpy很好的协作。tensorflow数学运算可以将python对象和numpy数组转换为tf.tensor对象。
    而tf.tensor.numpy()方法可以将对象的值作为numpy返回ndarray。
    2.0中默认使用eager模式，执行tf.executing_eagerly()来判断是否在eager模式下。
    :return:
    '''
#     矩阵相乘
    x = [[2,]]
    m = tf.matmul(x, x)
    print(type(x)) # tensorflow.python.framework.ops.EagerTensor
    print(type(m))
    print(m, m.numpy())    # 与numpy的ndarray不同的是：张量是不可变的对象，不光可以存储在内存当中也可以存储在GPU的显存当中
#     建立一个常量
    a = tf.constant([[1, 2], [3, 4]])
    print(type(a)) # tensorflow.python.framework.ops.EagerTensor
    print(a.numpy())
    b = tf.add(a, 1) # 相加
    c = tf.multiply(a, b) # 相乘
    print(b, c)
#     把一个值传换成tensor对象
    num = tf.convert_to_tensor(10)
    print(num)
#     python控制流
    for i in range(num.numpy()):
        i = tf.constant(i)  # 传换成tensor对象
        if int(i%2) == 0: # 会自动转换成python对象
            print('even')
        else:
            print('odd')
#   tensor+ndarray会得到tensor对象
    d = np.array([[5, 6], [7, 8]])
    print(a + d, (a+d).numpy())

def variable_differential():
    '''
    变量与自动微分运算
    :return:
    '''
#     定义变量
    v = tf.Variable(2)
    print(type(v), v) # tensorflow.python.ops.resource_variable_ops.ResourceVariable
    print(type(v+1)) # tensorflow.python.framework.ops.EagerTensor
    print(v+1) # 得到一个tensor对象
#     修改变量值
    v.assign(5)
    print(type(v), v) # tensorflow.python.ops.resource_variable_ops.ResourceVariable
    v.assign_add(1)
    print(type(v), v)
    # 读取变量值，得到一个tensor对象
    print(v.read_value())
#     记录运算过程，求解微分/梯度
    w = tf.Variable(1.0)
    with tf.GradientTape() as t: # 使用梯度磁带(上下文管理器)记录运算过程，自动跟踪变量的变化
        loss = w * w
    grad = t.gradient(loss, w) # 求解梯度，求解loss对w的微分
    print(grad)
#     对常量的微分计算
    w = tf.constant(3.0)
    with tf.GradientTape() as t:
        t.watch(w) # 常量需要watch
        loss = w * w
    grad = t.gradient(loss, w)
    print(grad)
#   对于微分求解，要求变量或常量需要是float类型。
#   而且在同一次计算当中，GradientTape在调用一次gradient方法之后就会释放资源，如果要计算多个微分的话是不行的。
    w = tf.constant(3.0)
    with tf.GradientTape(persistent=True) as t: # 让GradientTape持久下来
        t.watch(w)
        y = w * w
        z = y * y
    print(t.gradient(y, w))
    print(t.gradient(z, y))


if __name__ == '__main__':
#     eager模式
#     eager_mode()
#     变量与自动微分运算
    variable_differential()
