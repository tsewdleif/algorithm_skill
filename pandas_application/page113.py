#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/16 0016 下午 22:49
# @Author : West Field
# @File : page113.py

'''
狗名字统计的数据上，使用次数最高的前几个名字是什么呢？
'''
import pandas as pd

df = pd.read_csv("./dogNames2.csv")
print(df.head())
print(df.info())

# 排序
df = df.sort_values(by="Count_AnimalName", ascending=False)
print(df.head(10))

# pandas取行或者列，方括号写数字表示取行，写字符串表示取列
print(df[:10])
print(df["Row_Labels"])
print(type(df["Row_Labels"]))
