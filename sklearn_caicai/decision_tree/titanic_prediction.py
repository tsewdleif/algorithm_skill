#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/6/17 0017 下午 19:05
# @Author : West Field
# @File : titanic_prediction.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np

pd.set_option('display.max_columns', 20)  # a就是你要设置显示的最大列数参数
pd.set_option('display.max_rows', 20)  # b就是你要设置显示的最大的行数参数
pd.set_option('display.width', 1000)  # x就是你要设置的显示的宽度，防止轻易换行


def prediction():
    # 数据集来自kaggle，训练集包括特征和标签，test数据集是没有标签的
    ## 读取数据
    data = pd.read_csv("./dataset/titanic_train_data.csv")
    print(data)
    # 探索数据
    print(data.info())
    print(data.head(10))
    # 筛选特征
    data.drop(['Cabin', 'Name', 'Ticket'], inplace=True, axis=1)  # 删除无用特征和缺失值过多的特征
    # 填充缺失数据
    data["Age"] = data["Age"].fillna(data["Age"].mean())
    # 删除有缺失值的行
    data = data.dropna()
    print(data.info())
    # 特征数值化，因为决策树只能接收数字特征
    labels = data["Embarked"].unique().tolist()
    data["Embarked"] = data["Embarked"].apply(lambda x: labels.index(x))  # 特征中的取值没有联系的可以直接使用数值编码
    data["Sex"] = (data["Sex"] == "male").astype("int")
    print(data)
    # 分离特征和标签
    x = data.iloc[:, data.columns != "Survived"]
    y = data.loc[:, "Survived"]
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.3)
    for i in [Xtrain, Xtest, Ytrain, Ytest]:
        i.index = range(i.shape[0])  # 改变数据索引，使其顺序化
    print(Xtrain)
    ## 开始建模
    clf = DecisionTreeClassifier(random_state=25)
    clf = clf.fit(Xtrain, Ytrain)
    score = clf.score(Xtest, Ytest)
    print(score)
    # 交叉验证看模型的准确率分数
    score = cross_val_score(clf, x, y, cv=10).mean()
    print(score)
    ## 调参
    tr = []
    te = []
    for i in range(10):
        clf = DecisionTreeClassifier(random_state=25, max_depth=i + 1, criterion="entropy")
        clf = clf.fit(Xtrain, Ytrain)
        score_tr = clf.score(Xtrain, Ytrain)
        score_te = cross_val_score(clf, x, y, cv=10).mean()
        tr.append(score_tr)
        te.append(score_te)
    print(max(te))
    # 画图
    plt.plot(range(1, 11), tr, color="red", label="train")
    plt.plot(range(1, 11), te, color="blue", label="test")
    plt.xticks(range(1, 11))
    plt.legend()
    plt.show()
    # 网格搜索同时调整多个参数，本质是枚举
    gini_threholds = np.linspace(0, 0.5, 20)  # 包含20个数的0到0.5的等差数列
    parameters = {"criterion": ("gini", "entropy"), "splitter": ("best", "random"),
                  "max_depth": [*range(1, 10)],
                  "min_samples_leaf": [*range(1, 50, 5)],
                  "min_impurity_decrease": gini_threholds}
    clf = DecisionTreeClassifier(random_state=25)
    GS = GridSearchCV(clf, parameters, cv=10)
    GS = GS.fit(Xtrain, Ytrain)
    print(GS.best_params_)
    print(GS.best_score_)  # 此时的最好结果，不一定有非网格搜索时的结果好，因为网格搜索考虑了很多其他的参数因素。


if __name__ == "__main__":
    # 泰坦尼克生存预测
    prediction()
