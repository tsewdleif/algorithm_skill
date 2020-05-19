#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/14 0014 下午 20:28
# @Author : West Field
# @File : page29.py

'''
假设大家在30岁的时候,根据自己的实际情况,统计出来了你和你同桌各自从11岁到30岁每年交的女(男)朋友的数量如列表a和b,请在一个图中绘制出该数据的折线图,以便比较自己和同桌20年间的差异,同时分析每年交女(男)朋友的数量走势

'''

import matplotlib.pyplot as plt
from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="./simsun.ttc")
y_1 = [1,0,1,1,2,4,3,2,3,4,4,5,6,5,4,3,3,1,1,1]
y_2 = [1,0,3,1,2,2,3,3,2,1,2,1,1,1,1,1,1,1,1,1]
x = range(11, 31)

# 设置图形大小
plt.figure(figsize=(20, 8), dpi=80)

plt.plot(x, y_1, label="自己", color="orange", linestyle=":")
plt.plot(x, y_2, label="同桌", color="#DB7093", linestyle="--")

# 设置x、y轴刻度
_xticks_labels = ["{}岁".format(i) for i in x]
plt.xticks(x, _xticks_labels, fontproperties=my_font)
plt.yticks(range(0, 9))

# 绘制网格
plt.grid(alpha=0.4)

# 添加图例，显示中文
plt.legend(prop=my_font, loc="upper left")

plt.show()
