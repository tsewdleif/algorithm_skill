#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/17 0017 下午 17:04
# @Author : West Field
# @File : page136.py

'''
索引和复合索引
'''
import pandas as pd
import numpy as np

df = pd.DataFrame(np.ones(8).reshape(2, 4), index=["A", "B"], columns=list("abcd"))
df.loc["A", "a"] = 100
print(df)
# 获取索引
print(df.index)
# 修改索引
df.index = ["j", "k"]
print(df)
# 重新设定索引
df2 = df.reindex(["j", "f"])
print(df2)
# 设定某一列为索引
df3 = df.set_index("a")
print(df3)
df4 = df.set_index("a", drop=False)  # 仍保留列a
print(df4)
# 取index的唯一值，这里的index可以重复
print(df3.index.unique(), len(df3.index.unique()), list(df3.index.unique()))

# 设定两列为复合索引
df5 = df.set_index(["a", "b"], drop=False)
print(df5)

# 设置复合索引
a = pd.DataFrame(
    {'a': range(7), 'b': range(7, 0, -1), 'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'], 'd': list("hjklmno")})
b = a.set_index(["c", "d"])
print("-"*100)
print(b)
print(b.loc["one"])
print(b.loc["one","j"])
print("-"*100)

# Series复合索引
c = b["a"]
print(type(c)) # Series类型
print(c)
print(c["one", "j"])
# DataFrame复合索引
c2 = b[["a"]]
print(type(c2))
print(c2)
print(c2.loc["one"].loc["j"])
print("1-"*100)

# 复合索引交换索引
d = a.set_index(["d","c"])["a"]
print(d)
d = d.swaplevel()# 交换索引
print(d)

