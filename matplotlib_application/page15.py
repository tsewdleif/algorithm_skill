#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/14 0014 下午 15:51
# @Author : West Field
# @File : page15.py

'''
假设一天中每隔两个小时(range(2,26,2))的气温(℃)分别是[15,13,14.5,17,20,25,26,26,27,22,18,15]

'''
import matplotlib.pyplot as plt

x = range(2, 26, 2)
y = [15, 13, 14.5, 17, 20, 25, 26, 26, 27, 22, 18, 15]

# 设置图片大小
plt.figure(figsize=(20, 8), dpi=80)

# 绘图
plt.plot(x, y)

# 设置x、y轴的刻度
_xtick_labels = [i/2 for i in range(4, 49)]
plt.xticks(_xtick_labels[::3])# 每3个刻度显示一次
plt.yticks(range(min(y), max(y) + 1))

# 保存
plt.savefig("./t1.png")
# 展示图像
plt.show()
