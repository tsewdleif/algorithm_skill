#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/6/17 0017 上午 9:51
# @Author : West Field
# @File : algo.py

from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np

pd.set_option('display.max_columns', 20)  # a就是你要设置显示的最大列数参数
pd.set_option('display.max_rows', 100)  # b就是你要设置显示的最大的行数参数
pd.set_option('display.width', 1000)  # x就是你要设置的显示的宽度，防止轻易换行


def classify_tree():
    ## 加载红酒数据集
    wine = load_wine()
    # print(type(wine), wine) # sklearn.utils.Bunch 字典，key是data, vale是target
    print(wine.data.shape, wine.data)
    print(wine.target)
    # 把特征和标签拼接成表
    data_table = pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
    print(data_table)
    # 特征名称、标签名称
    print(wine.feature_names, wine.target_names)
    # 划分数据集，train_test_split会随机划分数据集
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3, random_state=30)
    print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)
    ## 开始建模
    clf = tree.DecisionTreeClassifier(criterion="entropy",  # criterion用来设置不纯度的计算的，entropy是使用信息熵，gini是使用基尼系数
                                      random_state=30,  # 决策树会随机的选取特征建模，使用random_state随机种子来固定。
                                      splitter="random")  # splitter是用来控制决策树中的随机选项的，best是决策树分支时优先使用更重要的特征进行分支，random是随机选择特征分支。
    clf = clf.fit(Xtrain, Ytrain)  # 训练数据
    score = clf.score(Xtest, Ytest)  # 返回预测的准确率accuracy
    print(score)
    ## 画树
    feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素', '颜色强度', '色调', 'od280/od315稀释葡萄酒',
                    '脯氨酸']
    # dot_data = tree.export_graphviz(clf
    #                                 , out_file='tree.dot'
    #                                 , feature_names=feature_name
    #                                 , class_names=["琴酒", "雪莉", "贝尔摩德"]
    #                                 , filled=True
    #                                 , rounded=True
    #                                 )
    # with open("tree.dot", encoding='utf-8') as f:
    #     dot_graph = f.read()
    # graph = graphviz.Source(dot_graph.replace("helvetica", "FangSong"))
    # graph.view()
    # 决策树中使用到的特征的特征重要性
    print(clf.feature_importances_)
    print([*zip(feature_name, clf.feature_importances_)])
    ## 剪枝参数，防止过拟合
    score = clf.score(Xtrain, Ytrain)
    print(score)
    clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=30, splitter="random",
                                      max_depth=3,  # 树的最大深度
                                      # min_samples_leaf=10, # 一个节点在分支后的每个子节点都必须包括至少min_samples_leaf个训练样本
                                      # min_samples_split=25 # 一个节点至少包含min_samples_split个训练样本才允许被分支
                                      )
    clf = clf.fit(Xtrain, Ytrain)
    # dot_data = tree.export_graphviz(clf
    #                                 , out_file='tree.dot'
    #                                 , feature_names=feature_name
    #                                 , class_names=["琴酒", "雪莉", "贝尔摩德"]
    #                                 , filled=True
    #                                 , rounded=True
    #                                 )
    # with open("tree.dot", encoding='utf-8') as f:
    #     dot_graph = f.read()
    # graph = graphviz.Source(dot_graph.replace("helvetica", "FangSong"))
    # graph.view()
    print(clf.score(Xtest, Ytest))
    # 超参数学习，确认最优的剪枝参数
    test = []
    for i in range(10):
        clf = tree.DecisionTreeClassifier(max_depth=i + 1,
                                          criterion="entropy",
                                          random_state=30,
                                          splitter="random")
        clf = clf.fit(Xtrain, Ytrain)
        score = clf.score(Xtest, Ytest)
        test.append(score)
    plt.plot(range(1, 11), test, color="red", label="max_depth")
    plt.legend()
    plt.show()
    ## 重要属性和接口，决策树结构接受的数据至少是一个二维矩阵
    print(clf.apply(Xtest))  # apply返回每个测试样本所在的叶子节点的索引
    print(clf.predict(Xtest))  # predict返回每个测试样本的分类/回归结果


def regression_tree():
    ## 加载波士顿房价数据集
    boston = load_boston()
    # print(boston, boston.data.shape, boston.target.shape)
    ## 实例化回归模型
    regressor = DecisionTreeRegressor(random_state=0)
    ## 交叉验证
    score = cross_val_score(regressor, boston.data, boston.target, cv=10,
                            scoring="neg_mean_squared_error")  # 使用scoring衡量模型
    print(score)
    ## 一维回归的图像绘制
    # 生成随机数种子
    rng = np.random.RandomState()
    # 生成80行1列的二维数据，应为回归树接受至少二维以上的数据
    X = np.sort(5 * rng.rand(80, 1))  # rand生成0到1的数据
    y = np.sin(X).ravel()  # 标签y只能是一维的，使用ravel降维
    y[::5] += 3 * (0.5 - rng.rand(16))  # y的值每隔5个加上一个噪声
    # 画散点图
    plt.scatter(X, y, s=20, edgecolors="black", c="darkorange", label="data")
    plt.show()
    # 建立两个模型，观察效果对比
    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_1.fit(X, y)
    regr_2.fit(X, y)
    # 测试
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]  # np.newaxis是数据升维
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)
    # 画图
    plt.figure()
    plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
    plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 分类树
    classify_tree()
    # 回归树
    regression_tree()
