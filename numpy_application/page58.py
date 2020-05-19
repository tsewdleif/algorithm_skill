#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/15 0015 上午 10:17
# @Author : West Field
# @File : page58.py

'''
numpy创建数组（矩阵）
'''

import numpy as np
import random

# 使用numpy生成数组，得到ndarray的类型
t1 = np.array([1, 2, 3])
print(t1)
print(type(t1))

t2 = np.array(range(10))
print(t2)
print(type(t2))

t3 = np.arange(4, 10, 2)
print(t3)
print(type(t3))

# 当前数组里边所存放数据的类型
print(t3.dtype)  # int32表示32位

# 指定创建的数组的数据类型
t4 = np.array(range(1, 4), dtype="float64")
print(t4)
print(t4.dtype)

t5 = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
print(t5)
print(t5.dtype)

# 调整数据类型
t6 = t5.astype("int8")
print(t6)
print(t6.dtype)

# 修改浮点型的小数位数
t7 = np.array([random.random() for i in range(10)])
print(t7)
print(t7.dtype)

t8 = np.round(t7, 2)
print(t8)
