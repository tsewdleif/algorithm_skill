#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/18 0018 下午 19:00
# @Author : West Field
# @File : page148_2.py

'''
统计出911数据中不同类型不同月份的电话的次数的变化情况
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./911.csv")
df["timeStamp"] = pd.to_datetime(df["timeStamp"])

# 添加列，表示分类
temp_list = df["title"].str.split(":").tolist()
cate_list = [i[0] for i in temp_list]
df["cate"] = pd.DataFrame(np.array(cate_list).reshape((df.shape[0], 1)))

# 设置索引
df.set_index("timeStamp", inplace=True)
print(df.head(1))

plt.figure(figsize=(20, 8), dpi=80)

# 分组
for group_name, group_data in df.groupby(by="cate"):
    # 对不同的分类都进行绘图。(group_name,group_data)是一个二元组，group_data是DataFrame类型
    # 重采样
    count_by_month = group_data.resample("M").count()["title"]
    # 绘图
    _x = count_by_month.index
    _y = count_by_month.values

    _x = [i.strftime("%Y%m%d") for i in _x]  # 格式化时间

    plt.plot(range(len(_x)), _y, label=group_name)

plt.xticks(range(len(_x)), _x, rotation=45)
plt.legend()
plt.show()
