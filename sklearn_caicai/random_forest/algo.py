#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/6/18 0018 上午 11:16
# @Author : West Field
# @File : algo.py

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import sklearn

pd.set_option('display.max_columns', 20)  # a就是你要设置显示的最大列数参数
pd.set_option('display.max_rows', 100)  # b就是你要设置显示的最大的行数参数
pd.set_option('display.width', 1000)  # x就是你要设置的显示的宽度，防止轻易换行


def random_forest_classify():
    ## 加载数据集
    wine = load_wine()
    print(wine.data.shape, wine.data.shape)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)
    ## 使用决策树和随机森林分别建模
    clf = DecisionTreeClassifier(random_state=0)
    rfc = RandomForestClassifier(random_state=0)
    clf = clf.fit(Xtrain, Ytrain)
    rfc = rfc.fit(Xtrain, Ytrain)
    score_c = clf.score(Xtest, Ytest)
    score_r = rfc.score(Xtest, Ytest)
    print("Single Tree:{}".format(score_c), "Random Forest:{}".format(score_r))
    ## 交叉验证
    # 一次交叉验证对比结果
    clf = DecisionTreeClassifier()
    clf_s = cross_val_score(clf, wine.data, wine.target, cv=10)
    rfc = RandomForestClassifier(n_estimators=25)  # n_estimators默认是100
    rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10)
    plt.plot(range(1, 11), clf_s, label="DecisionTree")
    plt.plot(range(1, 11), rfc_s, label="RandomForest")
    plt.legend()
    plt.show()
    # 十次交叉验证观察对比结果
    clf_l = []
    rfc_l = []
    for i in range(10):
        clf = DecisionTreeClassifier()
        clf_s = cross_val_score(clf, wine.data, wine.target, cv=10).mean()  # 取十次交叉验证的平均值
        clf_l.append(clf_s)
        rfc = RandomForestClassifier(n_estimators=25)
        rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10).mean()
        rfc_l.append(rfc_s)
    plt.plot(range(1, 11), clf_l, label="DecisionTree")
    plt.plot(range(1, 11), rfc_l, label="RandomForest")
    plt.legend()
    plt.show()  # 会发现决策树和随机森林的增减趋势基本同步
    ## 观察随机森林在不同n_estimators个数的准确率
    superpa = []
    for i in range(200):
        rfc = RandomForestClassifier(n_estimators=i + 1, n_jobs=-1)
        rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10).mean()
        superpa.append(rfc_s)
    print(max(superpa), superpa.index(max(superpa)))
    plt.figure(figsize=[20, 5])
    plt.plot(range(1, 201), superpa)
    plt.show()
    ## 随机森林分类器RFC的属性
    rfc = RandomForestClassifier(n_estimators=20, random_state=2)
    rfc = rfc.fit(Xtrain, Ytrain)
    # 查看随机森林中树的状况
    print(rfc.estimators_[0].random_state)  # 查看第0棵树的random_state
    print(rfc.estimators_[0].max_depth)  # 查看第0棵树的max_depth
    for i in range(len(rfc.estimators_)):
        print(rfc.estimators_[i].random_state)
    ## 使用包外数据作为验证集，无需划分训练集和测试集
    rfc = RandomForestClassifier(n_estimators=25, oob_score=True)
    rfc = rfc.fit(wine.data, wine.target)
    # 重要属性oob_score_
    print(rfc.oob_score_)
    ## 重要属性和接口
    rfc = RandomForestClassifier(n_estimators=25)
    rfc = rfc.fit(Xtrain, Ytrain)
    print(rfc.score(Xtest, Ytest))  # 在测试集上的准确率
    print(rfc.feature_importances_)  # 特征重要性
    print(rfc.apply(Xtest))  # 返回每个样本被分到的叶节点的索引
    print(rfc.predict(Xtest))  # 返回模型的预测值
    print(rfc.predict_proba(Xtest))  # 返回预测的结果属于每个类的概率值


def random_forest_regression():
    ## 加载波士顿房价预测的数据集
    boston = load_boston()
    print(boston)
    ## 建模
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    score = cross_val_score(regressor, boston.data, boston.target, cv=10,
                            scoring="neg_mean_squared_error")  # scoring是交叉验证的评价指标，默认是R^2
    print(score)
    # 查看sklearn中的模型评估指标
    print(sorted(sklearn.metrics.SCORERS.keys()))


if __name__ == "__main__":
    # 随机森林分类树
    # random_forest_classify()
    # 随机森林回归树
    random_forest_regression()
