#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/15 0015 下午 15:30
# @Author : West Field
# @File : page61.py

'''
数组的形状
'''

import numpy as np

# 查看数组形状
t1 = np.arange(12)
print(t1.shape)

t2 = np.array([[1,2,3], [4,5,6]])
print(t2.shape)

t3 = np.array([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]])
print(t3.shape)

# 修改数组形状
t4 = np.arange(12)
print(t4.reshape((3,4))) # 有返回值的，一般不会对原对象发生改变，这里t4在reshape之后不变

t5 = np.arange(24).reshape((2,3,4)) # 2快3行4列
print(t5)
print(t5.reshape((4,6)))

# 转换为1维数据
print(t5.reshape(24,))
print(t5.reshape((t5.shape[0]*t5.shape[1]*t5.shape[2],)))
print(t5.flatten())

