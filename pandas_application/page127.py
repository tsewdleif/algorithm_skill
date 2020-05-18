#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/17 0017 上午 10:53
# @Author : West Field
# @File : page127.py

'''
数据合并之join、merge
'''
import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.ones((2, 4)), index=["A", "B"], columns=list("abcd"))
df2 = pd.DataFrame(np.zeros((3, 3)), index=["A", "B", "C"], columns=list("xyz"))
df3 = pd.DataFrame(np.arange(9).reshape((3, 3)), columns=list("fax"))
df1.loc["A", "a"] = 100
print(df1)
print(df2)
print(df3)

# join：默认情况下他是把行索引相同的数据合并到一起
print(df1.join(df2))
print(df2.join(df1))

# merge：按照指定的列把数据按照一定的方式合并到一起
print(df1.merge(df3, on="a", how="inner"))  # 对于a的值相等的记录进行列的合并，默认内连接
print(df1.merge(df3, on="a", how="outer"))  # 外连接
print(df1.merge(df3, on="a", how="left"))  # 左连接
print(df1.merge(df3, on="a", how="right"))  # 右连接
