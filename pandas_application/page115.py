#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/17 0017 上午 7:24
# @Author : West Field
# @File : page115.py

'''
pandas之索引、切片
'''
import pandas as pd
import numpy as np

t = pd.DataFrame(np.arange(12).reshape(3, 4), index=list("abc"), columns=list("WXYZ"))

# loc：通过标签索引取数据
print(t.loc["a", "Z"])
print(type(t.loc["a", "Z"]))
print(t.loc[["a", "b"], :])
print(t.loc["a":"c", ["W", "Z"]])

# iloc：通过位置获取数据
print(t.iloc[1, :])
print(t.iloc[[0, 2], [2, 1]])
t.iloc[1:, :2] = np.nan
print(t)

# bool索引
print(t[(t["Y"] > 4) & (t["Y"] < 7)])

d = [{"name": "xiaoli", "age": 32, "tel": 10010}, {"name": "xiaogang/xiaoniao", "tel": 10000},
     {"name": "xiaowang/xiaohua/xiaohuan", "age": 32}]
t2 = pd.DataFrame(d)
print(t2[(t2["name"].str.len() > 4) & (t2["name"].str.len() < 7)])
print(t2["name"].str.split("/").tolist())
