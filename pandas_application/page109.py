#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/16 0016 下午 21:48
# @Author : West Field
# @File : page109.py

'''
pandas之DataFrame
'''
import numpy as np
import pandas as pd

t = pd.DataFrame(np.arange(12).reshape(3, 4))
print(t)

# 指定行索引和列索引
tt = pd.DataFrame(np.arange(12).reshape(3, 4), index=list("abc"), columns=list("WXYZ"))
print(tt)

# 字典方式建立DataFrame，两种方式
d1 = {"name": ["xiaoming", "xiaogang"], "age": [20, 32], "tel": [10086, 10010]}
t1 = pd.DataFrame(d1)
print(type(t1), t1)

d2 = [{"name": "xiaohong", "age": 32, "tel": 10010}, {"name": "xiaogang", "tel": 10000},
      {"name": "xiaowang", "age": 32}]
t2 = pd.DataFrame(d2)
print(type(t2), t2)
