#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/18 0018 下午 18:29
# @Author : West Field
# @File : page143.py

'''
不管在什么行业，时间序列都是一种非常重要的数据形式，很多统计数据以及数据的规律也都和时间序列有着非常重要的联系
而且在pandas中处理时间序列是非常简单的
'''
import pandas as pd

df = pd.date_range(start="20171230", end="20200531", freq="10D")
print(df)

# 生成从start开始的频率为freq的periods个时间索引
df2 = pd.date_range(start="20171230", periods=10, freq="10D")
print(df2)
