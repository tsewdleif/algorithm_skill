#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/16 0016 下午 16:28
# @Author : West Field
# @File : page96.py

'''
随机数
'''
import numpy as np

# 随机数
print(np.random.randint(10, 20, (4, 5)))  # 4行5列的数组，数组元素值范围在10到20

# 随机种子：可以通过设计相同的随机数种子，使得每次生成相同的随机数
np.random.seed(10)
print(np.random.randint(10, 20, (4, 5)))
np.random.seed(10)
print(np.random.randint(10, 20, (4, 5)))

# copy和view
a = np.arange(3)
b = a  # a、b的地址一样
print(id(a), id(b))
b = a[:]  # 视图的操作，会创建新的对象b，但是b的数据由a保管，它们数据变化保持一致
print(id(a), id(b))
b = a.copy()  # a、b数据互不影响
print(id(a), id(b))
