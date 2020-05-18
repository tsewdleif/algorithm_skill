#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/17 0017 上午 11:29
# @Author : West Field
# @File : page130.py

'''
现在我们有一组关于全球星巴克店铺的统计数据，如果我想知道美国的星巴克数量和中国的哪个多，或者我想知道中国每个省份星巴克的数量的情况，那么应该怎么办？

'''
import pandas as pd

file_path = "./starbucks_store_worldwide.csv"

df = pd.read_csv(file_path)
print(df.head(1))
print(df.info())

# 按照国家字段分组，得到DataFrameGroupBy类型对象
grouped = df.groupby(by="Country")
print(type(grouped))
print(grouped)

# 进行遍历
# for i in grouped:
#     print(i)  # 输出元组，第一个值是国家
#     print("*" * 100)
#     print(i[1], type(i[1]))  # 输出DataFrame
#     print("*" * 100)

# 调用聚合方法
print(grouped.count())  # 统计所有列
country_count = grouped["Brand"].count()  # 统计Brand列
print(country_count["US"])  # 美国星巴克店铺数量
print(country_count["CN"])  # 中国星巴克店铺数量

# 统计中国每个省店铺的数量
china_data = df[df["Country"] == "CN"]
grouped = china_data.groupby(by="State/Province").count()["Brand"]
print(grouped)

# 数据按照多个条件进行分组
grouped1 = df["Brand"].groupby(by=[df["Country"], df["State/Province"]]).count()
print(type(grouped1))  # Series类型，前两列是索引，后一列是数据。因为上边的df["Brand"]是Series类型
print(grouped1)
grouped2 = df[["Brand"]].groupby(by=[df["Country"], df["State/Province"]]).count()
print(type(grouped2))  # DataFrame类型，因为上边df[["Brand"]]是DataFrame类型
print(grouped2)

# 复合索引
print(grouped1.index)
