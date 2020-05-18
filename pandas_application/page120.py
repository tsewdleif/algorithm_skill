#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/17 0017 上午 10:06
# @Author : West Field
# @File : page120.py

'''
缺失数据的处理
'''
import pandas as pd

d = [{"name": "xiaoli", "age": 32, "tel": 10010}, {"name": "xiaogang", "tel": 10000},
     {"name": "xiaogang", "age": 28}]
t = pd.DataFrame(d)

# 判定是否是NaN
print(pd.isna(t))
print(pd.notna(t))

# 选择t中的age列非空的数据
print(t[pd.notna(t["age"])])

# 删除有NaN的行数据
print(t.dropna(axis=0, how="any"))
# 删除全部为NaN的行
print(t.dropna(axis=0, how="all"))

# 原地修改
t2 = t.copy()
print(t2)
print(t2.dropna(axis=0, how="any", inplace=True))
print(t2)

# 填充NaN为列的均值，如果填充为0，则均值会变小，NaN不会计入均值的分母计算中
print(t.fillna(t.mean()))
t["age"] = t["age"].fillna(t["age"].mean())  # 只填充age列
print(t)

# 输出一列的所有值
print(t["name"].tolist())
print(t["name"].unique())

# 获取最值以及位置
print(t["age"].max())
print(t["age"].idxmax())
