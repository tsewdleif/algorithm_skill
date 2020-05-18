#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/18 0018 下午 17:34
# @Author : West Field
# @File : page142.py

'''
现在我们有2015到2017年25万条911的紧急电话的数据，请统计出出这些数据中不同类型的紧急情况的次数，如果我们还想统计出不同月份不同类型紧急电话的次数的变化情况，应该怎么做呢?
'''
import pandas as pd
import numpy as np

df = pd.read_csv("./911.csv")

print(df.head(10))
print(df.info())

# 获取分类
temp_list = df["title"].str.split(": ").tolist()
cate_list = [i[0] for i in temp_list]
df["cate"] = pd.DataFrame(np.array(cate_list).reshape((df.shape[0], 1)))

print(df.head(5))
print(df.groupby(by="cate").count()["title"])
