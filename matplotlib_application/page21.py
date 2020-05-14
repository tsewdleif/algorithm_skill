#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/14 0014 下午 16:23
# @Author : West Field
# @File : page21.py

'''
如果列表a表示10点到12点的每一分钟的气温,如何绘制折线图观察每分钟气温的变化情况?
a= [random.randint(20,35) for i in range(120)]
'''
import matplotlib.pyplot as plt
from matplotlib import font_manager
import random

# matplotlib默认不支持中文字符，设置中文字体
my_font = font_manager.FontProperties(fname="./simsun.ttc")
# my_font = font_manager.FontProperties(fname="C:\Windows\Fonts\STSONG.TTF")

x = range(0, 120)
y = [random.randint(20, 35) for i in range(120)]

plt.figure(figsize=(20,8),dpi=80)

plt.plot(x,y)

# 调整x、y轴的刻度
_xtick_labels = ["10点{}分".format(i) for i in range(60)]
_xtick_labels += ["11点{}分".format(i) for i in range(60)]
# 取步长，保持前两个参数长度一致，即用第二个参数代替第一个数字刻度的每个位置
plt.xticks(list(x)[::3], _xtick_labels[::3], rotation=45, fontproperties=my_font)

# 添加描述信息
plt.xlabel("时间", fontproperties=my_font)
plt.ylabel("温度 单位(℃)", fontproperties=my_font)
plt.title("10点到12点每分钟的气温变化情况", fontproperties=my_font)


plt.show()


