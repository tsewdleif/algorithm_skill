#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/15 0015 下午 16:51
# @Author : West Field
# @File : page74.py

'''
现在这里有一个英国和美国各自youtube1000多个视频的点击,喜欢,不喜欢,评论数量(["views","likes","dislikes","comment_total"])的csv,运用刚刚所学习的知识,我们尝试来对其进行操作
'''

import numpy as np

# numpy读取数据
us_file_path = "./youtube_video_data/US_video_data_numbers.csv"
uk_file_path = "./youtube_video_data/GB_video_data_numbers.csv"

# delimiter数据的分割方式，dtype读取的数据以什么格式呈现，unpack表示转置
t1 = np.loadtxt(us_file_path, delimiter=",", dtype="int", unpack=True)
t2 = np.loadtxt(uk_file_path, delimiter=",", dtype="int")

print(t1)
print(t2)

# 二维数组转置
t3 = np.arange(24).reshape((4, 6))
print(t3)
print(t3.transpose())
print(t3.T)
print(t3.swapaxes(1, 0))  # 交换轴

# numpy索引和切片
print('*' * 100)
# 取行
print(t2[2])
print(t2[2:])
print(t2[[2, 8, 10]])
# 取列
print(t2[1, :])  # 取第二行所有列
print(t2[:, 2:])
# 取多个不相邻的点，取(0,0)，(2,1)，(2,3)
print(t2[[0, 2, 2], [0, 1, 3]])
