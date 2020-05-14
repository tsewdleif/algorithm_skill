#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/14 0014 下午 21:49
# @Author : West Field
# @File : page43.py

'''
假设你知道了列表a中电影分别在2017-09-14(b_14), 2017-09-15(b_15), 2017-09-16(b_16)三天的票房,为了展示列表中电影本身的票房以及同其他电影的数据对比情况,应该如何更加直观的呈现该数据?
'''

import matplotlib.pyplot as plt
from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="./simsun.ttc")

a = ["猩球崛起3：终极之战", "敦刻尔克", "蜘蛛侠：英雄归来", "战狼2"]
b_16 = [15746, 312, 4497, 319]
b_15 = [12357, 156, 2045, 168]
b_14 = [2358, 399, 2358, 362]

# 条形图的宽度
bar_width = 0.2

# x坐标列表
x_14 = list(range(len(a)))
x_15 = [i + bar_width for i in x_14]
x_16 = [i + 2 * bar_width for i in x_14]

# 设置图形大小
plt.figure(figsize=(20, 8), dpi=80)

# 绘制条形图
plt.bar(range(len(a)), b_14, width=0.2, label="9月14日")
plt.bar(x_15, b_15, width=bar_width, label="9月15日")
plt.bar(x_16, b_16, width=bar_width, label="9月16日")

# 设置图例
plt.legend(prop=my_font)

# 设置x轴的刻度
plt.xticks(x_15, a, fontproperties=my_font)

plt.show()
