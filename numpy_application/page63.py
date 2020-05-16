#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/15 0015 下午 15:45
# @Author : West Field
# @File : page63.py

'''
数组的计算
'''

import numpy as np

# 数组和数的计算，广播机制：数组中每个数和数字运算
t1 = np.arange(24).reshape((4, 6))
print(t1 + 2)
print(t1 / 0)  # 0/0得nan，即非数字。数字/0得inf，即无限大的数。

# 形状一样的的运算是对应位置元素的计算
t2 = np.arange(100, 124).reshape((4, 6))
print(t1 + t2)
print(t1 * t2)

# 形状不同的数组之间的计算，在形状一致的维度上运算
t3 = np.arange(0, 6)
print(t1 - t3)

t4 = np.arange(4).reshape((4, 1))
print(t1 - t4)

t5 = np.arange(10)
# print(t1-t5) # 会报错，行列都不同
