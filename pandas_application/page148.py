#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/18 0018 下午 18:41
# @Author : West Field
# @File : page148.py

'''
统计出911数据中不同月份电话次数的变化情况

'''
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./911.csv")

# 把字符串转化为pandas中的datetime类型
df["timeStamp"] = pd.to_datetime(df["timeStamp"])

# 设置索引
df.set_index("timeStamp", inplace=True)

print(df.head())

# 重采样：指的是将时间序列从一个频率转化为另一个频率进行处理的过程，将高频率数据转化为低频率数据为降采样，低频率转化为高频率为升采样
# pandas提供了一个resample的方法来帮助我们实现频率转化
count_by_month = df.resample("M").count()["title"]
print(count_by_month)

_x = count_by_month.index
_y = count_by_month.values

# 格式化_x
# 查看_x的所有方法，找到格式化的方法
for i in _x:
    print(dir(i))
    break
_x = [i.strftime("%Y%m%d") for i in _x]

plt.figure(figsize=(20, 8), dpi=80)
plt.plot(range(len(_x)), _y)
plt.xticks(range(len(_x)), _x, rotation=45)
plt.show()
