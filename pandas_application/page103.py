#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/16 0016 下午 20:41
# @Author : West Field
# @File : page103.py

'''
pandas之Series创建
'''

import pandas as pd
import string

t = pd.Series([1, 2, 31, 12, 3, 4])
print(type(t))
print(t)

# 指定索引
t2 = pd.Series([1, 23, 2, 2, 1], index=list("abcde"))
print(t2)

# 通过字典创建Series，默认索引是字典的键
temp_dict = {"name": "xiaohong", "age": 30, "tel": 10086}
t3 = pd.Series(temp_dict)
print(t3)

a = {string.ascii_uppercase[i]: i for i in range(10)}
print(a)
# 重新给其指定索引，如果能够对应上就取其值，否则就为NaN
t4 = pd.Series(a, index=list(string.ascii_uppercase[5:15]))
print(t4)
