#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/16 0016 下午 16:51
# @Author : West Field
# @File : page87.py

'''
t中存在nan值，如何操作把其中的nan填充为每一列的均值

'''
import numpy as np


def fill_ndarray(t):
    for i in range(t.shape[1]):  # 遍历每一列
        temp_col = t[:, i]  # 当前的一列
        nan_num = np.count_nonzero(temp_col != temp_col)
        if nan_num != 0:  # 不为0，说明当前这一列中有nan
            tempp_not_nan_col = temp_col[temp_col == temp_col]  # 当前一列不为nan的array
            # 选中当前为nan的位置，把值赋值为不为nan的均值
            temp_col[np.isnan(temp_col)] = tempp_not_nan_col.mean()
    return t


if __name__ == "__main__":
    t = np.arange(24).reshape((6, 4)).astype("float")
    t[1, 2:] = np.nan
    print(t)
    t = fill_ndarray(t)
    print(t)
