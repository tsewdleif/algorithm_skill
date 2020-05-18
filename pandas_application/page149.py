#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/18 0018 下午 19:27
# @Author : West Field
# @File : page149.py

'''
现在我们有北上广、深圳、和沈阳5个城市空气质量数据，请绘制出5个城市的PM2.5随时间的变化情况
'''
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./PM2.5/BeijingPM20100101_20151231.csv")

# 之前所学习的DatetimeIndex可以理解为时间戳，那么现在我们要学习的PeriodIndex可以理解为时间段
# 把分开的时间字符串通过periodIndex的方法转化为pandas的时间类型
period = pd.PeriodIndex(year=df["year"], month=df["month"], day=df["day"], hour=df["hour"], freq="H")
df["datetime"] = period  # 增加一列
print(df.head(10))

# 把datetime设置为索引
df.set_index("datetime", inplace=True)

# 进行降采样
df = df.resample("7D").mean()

# 处理缺失数据，删除缺失数据
data = df["PM_US Post"].dropna()

# 画图
_x = data.index

_y = data.values

plt.figure(figsize=(20, 8), dpi=80)
plt.plot(range(len(_x)), _y)
plt.xticks(range(0, len(_x), 10), list(_x)[::10], rotation=45)
plt.show()
