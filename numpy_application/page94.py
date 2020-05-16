#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/16 0016 下午 16:09
# @Author : West Field
# @File : page94.py

'''
数组的拼接
'''
import numpy as np

us_data = "./youtube_video_data/US_video_data_numbers.csv"
uk_data = "./youtube_video_data/GB_video_data_numbers.csv"

# 加载国家数据
us_data = np.loadtxt(us_data, delimiter=",", dtype=int)
uk_data = np.loadtxt(uk_data, delimiter=",", dtype=int)

# 交换两行数据
print(us_data)
us_data[[1, 2], :] = us_data[[2, 1], :]
print(us_data)
us_data[:, [0, 2]] = us_data[:, [2, 0]]
print(us_data)

# 构造全为0、1的数组
zeros_data = np.zeros((us_data.shape[0], 1)).astype(int)
ones_data = np.ones((uk_data.shape[0], 1)).astype(int)

# 分别添加一列全为0、1的数组
us_data = np.hstack((us_data, zeros_data))
uk_data = np.hstack((uk_data, ones_data))

# 拼接两组数据
final_data = np.vstack((us_data, uk_data))
print(final_data)

# 创建一个E矩阵
print(np.eye(10))

# 取最大值的位置
print(np.argmax(us_data, axis=1))
