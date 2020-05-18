#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/18 0018 下午 17:12
# @Author : West Field
# @File : page141.py

'''
现在我们有全球排名靠前的10000本书的数据，那么请统计一下不同年份书的平均评分情况
'''
import pandas as pd
import matplotlib.pyplot as plt

file_path = "./books.csv"
df = pd.read_csv(file_path)

# 取出列中为Nan的行
data = df[pd.notna(df["original_publication_year"])]

grouped = df["average_rating"].groupby(by=data["original_publication_year"]).mean()

print(grouped)

_x = grouped.index
_y = grouped.values

plt.figure(figsize=(20, 8), dpi=80)

plt.plot(range(len(_x)), _y)

plt.xticks(list(range(len(_x)))[::10], _x[::10].astype("int"), rotation=45)

plt.show()
