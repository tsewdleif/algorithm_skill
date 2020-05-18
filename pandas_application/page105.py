#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/16 0016 下午 21:11
# @Author : West Field
# @File : page105.py

'''
pandas之切片和索引
'''
import pandas as pd

temp_dict = {"name": "xiaohong", "age": 30, "tel": 10086}
t = pd.Series(temp_dict)

# 通过索引取值
print(t["age"])
# 通过位置取值
print(t[1])

# 取多行数据
print(t[["age", "tel"]])
print(t[[1, 2]])

# bool索引
t1 = pd.Series([1, 2, 31, 12, 3, 4])
print(t1[t1 > 10])

# 取索引和值
index = t.index
print(type(index), index)
for i in t.index:
    print(i)
values = t.values
print(type(values), values)
for i in values:
    print(i)
