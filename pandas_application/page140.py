#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/18 0018 下午 16:41
# @Author : West Field
# @File : page140.py

'''
使用matplotlib呈现出每个中国每个城市的店铺数量
'''
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="../matplotlib_application/simsun.ttc")
file_path = "./starbucks_store_worldwide.csv"

df = pd.read_csv(file_path)
df = df[df["Country"] == "CN"]

data = df.groupby(by="City").count()["Brand"].sort_values(ascending=False)[:25]

_x = data.index
_y = data.values

plt.figure(figsize=(20, 12), dpi=80)

plt.barh(range(len(_x)), _y, height=0.3, color="orange")

plt.yticks(range(len(_x)), _x, fontproperties=my_font)

plt.show()
