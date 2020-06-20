#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/6/18 0018 下午 23:14
# @Author : West Field
# @File : fill_in_missing_values.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.impute import SimpleImputer  # 填补缺失值的类
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_columns', 10)  # a就是你要设置显示的最大列数参数
pd.set_option('display.max_rows', 10)  # b就是你要设置显示的最大的行数参数
pd.set_option('display.width', 100)  # x就是你要设置的显示的宽度，防止轻易换行


def fill_in():
    ## 加载数据
    dataset = load_boston()
    print(dataset.data.shape)
    # 总共506*13个数据
    X_full, y_full = dataset.data, dataset.target
    n_samples = X_full.shape[0]  # 行数
    n_features = X_full.shape[1]  # 列数
    # 假设要设置为50%的数据为缺失值
    rng = np.random.RandomState(0)  # 确定一个随机数种子
    missing_rate = 0.5  # 缺失率
    n_missing_sample = int(np.floor(n_samples * n_features * missing_rate))  # 缺失值的个数
    print(n_missing_sample)
    # 缺失的数据要分布在各行各列，需要一个行号和列号来确定一个位置
    missing_features = rng.randint(0, n_features, n_missing_sample)  # 从0到n_features随机挑选n_missing_sample个整数
    missing_samples = rng.randint(0, n_samples, n_missing_sample)
    # missing_samples = rng.choice(n_samples, n_missing_sample, replace=False) # 可以从0到n_samples中选择n_missing_sample个无重复的数
    print(len(missing_features), len(missing_samples))
    # 创造缺失数据集
    X_missing = X_full.copy()
    y_missing = y_full.copy()
    X_missing[missing_samples, missing_features] = np.nan
    X_missing = pd.DataFrame(X_missing)
    print(X_missing)
    ## 数据填充
    # 使用均值进行填充
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_missing_mean = imp_mean.fit_transform(X_missing)
    print(X_missing_mean)
    print(pd.DataFrame(X_missing_mean).isnull().sum())
    # 使用0进行填补
    imp_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    X_missing_0 = imp_0.fit_transform(X_missing)
    print(X_missing_0)
    print(pd.DataFrame(X_missing_0).isnull().sum())
    # 使用随机森林回归填补缺失值。从缺失值数量最小的特征开始填充，因为训练数据越多，回归预测填充的效果越好。其它特征缺失值先用0进行填充。
    X_missing_reg = X_missing.copy()
    sortindex = np.argsort(X_missing_reg.isnull().sum(axis=0)).values
    print(sortindex)
    # 先使用0填充非随机森林要填充特征的缺失值
    for i in sortindex:
        df = X_missing_reg
        fillc = df.iloc[:, i]
        df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(y_full)], axis=1)
        df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)  # 填充0
        # 划分训练集和测试集
        Ytrain = fillc[fillc.notnull()]
        Ytest = fillc[fillc.isnull()]
        Xtrain = df_0[Ytrain.index, :]
        Xtest = df_0[Ytest.index, :]
        # 用随机森林回归来填补缺失值
        rfc = RandomForestRegressor(n_estimators=100)
        rfc = rfc.fit(Xtrain, Ytrain)
        Ypredict = rfc.predict(Xtest)
        # 将填充好的特征返回到我们的原始特征矩阵中
        X_missing_reg.loc[X_missing_reg.iloc[:, i].isnull(), i] = Ypredict
    print(X_missing_reg)
    print(X_missing_reg.isnull().sum())
    ## 对补充好的数据建模打分，取MSE0结果
    X = [X_full, X_missing_mean, X_missing_0, X_missing_reg]
    mse = []
    for x in X:
        estimator = RandomForestRegressor(random_state=0, n_estimators=100)
        scores = cross_val_score(estimator, x, y_full, scoring='neg_mean_squared_error', cv=5).mean()
        mse.append(scores * -1)
    print(mse)  # MSE越小越好
    ## 用得到的结果画条形图
    x_labels = ['Full data',
                'Zero Imputation',
                'Mean Imputation',
                'Regressor Imputation']
    colors = ['r', 'g', 'b', 'orange']
    plt.figure(figsize=(12, 6))  # 画出画布
    ax = plt.subplot(111)  # 添加子图
    for i in np.arange(len(mse)):
        ax.barh(i, mse[i], color=colors[i], alpha=0.6, align='center')
    ax.set_title('Imputation Techniques with Boston Data')
    ax.set_xlim(left=np.min(mse) * 0.9,
                right=np.max(mse) * 1.1)
    ax.set_yticks(np.arange(len(mse)))
    ax.set_xlabel('MSE')
    ax.set_yticklabels(x_labels)
    plt.show()


if __name__ == "__main__":
    # 缺失值填充
    fill_in()
