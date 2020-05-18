#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/16 0016 下午 22:28
# @Author : West Field
# @File : page112.py

'''
DataFrame的基础属性和整体情况查询
'''
import pandas as pd

d = [{"name": "xiaohong", "age": 32, "tel": 10010}, {"name": "xiaogang", "tel": 10000},
     {"name": "xiaowang", "age": 32}]
t = pd.DataFrame(d)
print(t)

# DataFrame的基本属性
print(t.index)
print(t.columns)
print(t.values)
print(t.shape)
print(t.dtypes)
print(t.ndim)

# DataFrame整体情况查询
print("*" * 100)
print(t.head())
print(t.tail())
print(t.info())  # 行列数目、内容占用大小等
print(t.describe())  # 统计信息
