#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/16 0016 下午 18:59
# @Author : West Field
# @File : page89.py

'''
希望了解英国的youtube中视频的评论数和喜欢数的关系，应该如何绘制改图

'''
import numpy as np
import matplotlib.pyplot as plt

uk_file_path = "./youtube_video_data/GB_video_data_numbers.csv"

t_uk = np.loadtxt(uk_file_path, delimiter=",", dtype="int")

t_uk = t_uk[t_uk[:, 1] <= 500000]  # 取喜欢数目小于50万的数据。如果想知道更加相关的关系可以细分视频的种类观察相关性。

t_uk_comment = t_uk[:, -1]
t_uk_like = t_uk[:, 1]

plt.figure(figsize=(20, 8), dpi=80)
plt.scatter(t_uk_like, t_uk_comment)

plt.show()
