#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/16 0016 上午 10:35
# @Author : West Field
# @File : page79.py

'''
numpy中数值的修改
'''

import numpy as np

t = np.arange(24).reshape((4, 6))
# 索引切片赋值
t[:, 2:4] = 0  # 第三第四列赋值为0
print(t)

# 布尔索引赋值
print(t < 10)
t[t < 10] = 3  # 小于10的数字赋值为3
print(t)

# 三元运算符赋值
a = 1 if 4 > 5 else 8
print(a)
t = np.where(t < 5, 0, 20)
print(t)

# 裁剪
t = t.clip(10, 15)  # 将小于10的数字替换为10，大于15的数字替换为15
print(t)

# 将nan赋值给元素
t = t.astype(float)  # nan是float类型，所以先要把数组元素转化为float，才能将nan给其赋值
t[3, 3] = np.nan
print(t)

# 两个nan是不相等的
print(np.nan == np.nan)
# 统计数组中为0的个数
print(np.count_nonzero(t))
# 判定数组中nan的个数
print(np.count_nonzero(t != t))
# 判定一个元素是否是nan
print(np.isnan(t[3, 3]))
# 把是nan的元素赋值为0
t[np.isnan(t)] = 0
print(t)

# numpy中的统计函数
print(t.sum(axis=None), np.sum(t))
print(t.mean(axis=None), np.mean(t))
print(t.max(), t.min())
print(t.std())
