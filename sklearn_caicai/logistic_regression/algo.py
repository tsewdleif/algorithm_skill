#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/6/26 0026 下午 21:22
# @Author : West Field
# @File : algo.py

from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_iris


def lr():
    '''
    逻辑回归，其本质是由线性回归变化而来的，一种分类算法。
    线性回归用于解决回归预测问题，而逻辑回归用于解决分类问题，
    逻辑回归比线性回归多了一个映射函数，即Sigmoid函数，将线性回归的预测值映射到0到1之间，来解决二分类问题。
    线性回归是利用最小二乘法求解特征系数参数的一个模型，而逻辑回归是使用梯度下降求解特征系数参数。

    逻辑回归优点：
    逻辑回归是一个线性模型，对于特征和标签之间存在线性关系强的数据集效果较好，否则效果较差，在金融领域应用较多；
    计算速度较SVM和随机森林快；
    返回的分类结果是小数的概率形式，而非0、1。

    我们使用“损失函数”这个评估指标，来衡量参数为的模型拟合训练集时产生的信息损失的大小，并以此求出最优参数。
    注意：没有“求解参数”需求的模型没有损失函数，比如KNN，决策树，它们需要设置的都是超参数。
    逻辑回归的损失函数是由极大似然估计推导出来的。

    虽然逻辑回归和线性回归是天生欠拟合的模型，但我们还是需要  控制过拟合的技术来帮助我们调整模型，对逻辑回归中过拟合的控制，通过正则化来实现。
    通常来说，如果我们的主要目的只是为了防止过拟合，选择L2正则化就足够了。但是如果选择L2正则化后还是过拟合，模型在未知数据集上的效果表现很差，就可以考虑L1正则化。
    L1正则化越强，选出来的特征就越少，以此来防止过拟合。如果特征量很大，数据维度很高，我们会倾向于使用L1正则化。
    :return:
    '''
    ## 加载乳腺癌二分类数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    print(X.shape)
    ## L1正则化和L2正则化下的逻辑回归模型。
    lrl1 = LR(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)
    lrl2 = LR(penalty="l2", solver="liblinear", C=0.5, max_iter=1000)
    # L1正则化会筛选掉大量的特征，即有很多特征的参数为0
    lrl1 = lrl1.fit(X, y)
    print(lrl1.coef_)  # 查看训练后的模型的每个特征所对应的参数
    print((lrl1.coef_ != 0).sum(axis=1))
    # L2正则化会尽量的让每一个特征有用，即特征参数都不为0
    lrl2 = lrl2.fit(X, y)
    print(lrl2.coef_)
    print((lrl2.coef_ != 0).sum(axis=1))
    ## 究竟哪个正则化的效果更好呢？还是都差不多？
    l1 = []
    l2 = []
    l1test = []
    l2test = []
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
    for i in np.linspace(0.05, 1, 19):
        lrl1 = LR(penalty="l1", solver="liblinear", C=i, max_iter=1000)
        lrl2 = LR(penalty="l2", solver="liblinear", C=i, max_iter=1000)
        lrl1 = lrl1.fit(Xtrain, Ytrain)
        l1.append(accuracy_score(lrl1.predict(Xtrain), Ytrain))
        l1test.append(accuracy_score(lrl1.predict(Xtest), Ytest))
        lrl2 = lrl2.fit(Xtrain, Ytrain)
        l2.append(accuracy_score(lrl2.predict(Xtrain), Ytrain))
        l2test.append(accuracy_score(lrl2.predict(Xtest), Ytest))
    graph = [l1, l2, l1test, l2test]
    color = ["green", "black", "lightgreen", "gray"]
    label = ["L1", "L2", "L1test", "L2test"]
    plt.figure(figsize=(6, 6))
    for i in range(len(graph)):
        plt.plot(np.linspace(0.05, 1, 19), graph[i], color[i], label=label[i])
    plt.legend(loc=4)  # 4表示图例的位置在右下角
    plt.show()


def feature_engining_in_lr():
    '''
    逻辑回归中的特征工程
    :return:
    '''
    ## 加载乳腺癌二分类数据集
    data = load_breast_cancer()
    ## 逻辑回归模型
    LR_ = LR(solver="liblinear", C=0.9, random_state=420)
    print(cross_val_score(LR_, data.data, data.target, cv=10).mean())  # 0.9490601503759398
    ## 特征选择
    X_embedded = SelectFromModel(LR_, norm_order=1).fit_transform(data.data, data.target)
    print(X_embedded.shape)  # (569, 9)
    print(cross_val_score(LR_, X_embedded, data.target, cv=10).mean())  # 0.9368107769423559，特征维度下降了，准确率下降的不多
    ## 一旦调整threshold，就不是在使用L1正则化选择特征，而是使用模型的属性.coef_中生成的各个特征的系数来选择。
    # coef_虽然返回的是特征的系数，但是系数的大小和决策树中的feature_importances_以及降维算法中的可解释性方差explained_vairance_概念相似，
    # 其实都是衡量特征的重要程度和贡献度的，因此SelectFromModel中的参数threshold可以设置为coef_的阈值，即可以剔除系数小于threshold中输入的数字的所有特征。
    # fullx = []
    # fsx = []
    # threshold = np.linspace(0, abs((LR_.fit(data.data, data.target).coef_)).max(), 20)
    # k = 0
    # for i in threshold:
    #     X_embedded = SelectFromModel(LR_, threshold=i).fit_transform(data.data, data.target)
    #     fullx.append(cross_val_score(LR_, data.data, data.target, cv=5).mean())
    #     fsx.append(cross_val_score(LR_, X_embedded, data.target, cv=5).mean())
    #     print((threshold[k], X_embedded.shape[1]))
    #     k += 1
    # plt.figure(figsize=(20, 5))
    # plt.plot(threshold, fullx, label="full")
    # plt.plot(threshold, fsx, label="feature selection")
    # plt.xticks(threshold)
    # plt.legend()
    # plt.show()
    # 然而，以上这种方法其实是比较无效的，当threshold越来越大，被删除的特征越来越     多，模型的效果也越来越差。
    ## 第二种调整方法，是调逻辑回归的类LR_，通过画C的学习曲线来实现。
    # fullx = []
    # fsx = []
    # C = np.arange(0.01, 10.01, 0.5)
    # for i in C:
    #     LR_ = LR(solver="liblinear", C=i, random_state=420)
    #     fullx.append(cross_val_score(LR_, data.data, data.target, cv=10).mean())
    #     X_embedded = SelectFromModel(LR_, norm_order=1).fit_transform(data.data, data.target)
    #     fsx.append(cross_val_score(LR_, X_embedded, data.target, cv=10).mean())
    # print(max(fsx), C[fsx.index(max(fsx))]) # 0.9561090225563911 6.01
    # plt.figure(figsize=(20, 5))
    # plt.plot(C, fullx, label="full")
    # plt.plot(C, fsx, label="feature selection")
    # plt.xticks(C)
    # plt.legend()
    # plt.show()
    # 继续细化学习曲线
    fullx = []
    fsx = []
    C = np.arange(5.51, 6.51, 0.005)
    for i in C:
        LR_ = LR(solver="liblinear", C=i, random_state=420)
        fullx.append(cross_val_score(LR_, data.data, data.target, cv=10).mean())
        X_embedded = SelectFromModel(LR_, norm_order=1).fit_transform(data.data, data.target)
        fsx.append(cross_val_score(LR_, X_embedded, data.target, cv=10).mean())
    print(max(fsx), C[fsx.index(max(fsx))])  # 0.9561090225563911 5.834999999999993
    plt.figure(figsize=(20, 5))
    plt.plot(C, fullx, label="full")
    plt.plot(C, fsx, label="feature selection")
    plt.xticks(C)
    plt.legend()
    plt.show()
    # 验证模型效果：降维之前
    LR_ = LR(solver="liblinear", C=5.834999999999993, random_state=420)
    print(cross_val_score(LR_, data.data, data.target, cv=10).mean())  # 0.9508145363408522
    # 验证模型效果：降维之后
    LR_ = LR(solver="liblinear", C=5.834999999999993, random_state=420)
    X_embedded = SelectFromModel(LR_, norm_order=1).fit_transform(data.data, data.target)
    print(cross_val_score(LR_, X_embedded, data.target, cv=10).mean())  # 0.9561090225563911
    print(X_embedded.shape)  # (569, 11)
    # 准确率从0.9508到0.9561，这样我们就实现了在特征选择的前提下，保持模型拟合的高效。


def max_iter_parameter():
    '''
    在sklearn当中，我们设置参数max_iter最大迭代次数来代替步长，帮助我们控制模型的迭代速度并适时地让模型停下。
    max_iter越大，代表步长越小，模型迭代时间越长，反之，则代表步长设置很大，模型迭代时间很短。
    :return:
    '''
    ## 加载乳腺癌二分类数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    ## max_iter的学习曲线
    l2 = []
    l2test = []
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
    for i in np.arange(1, 201, 10):
        lrl2 = LR(penalty="l2", solver="liblinear", C=0.9, max_iter=i)
        lrl2 = lrl2.fit(Xtrain, Ytrain)
        l2.append(accuracy_score(lrl2.predict(Xtrain), Ytrain))
        l2test.append(accuracy_score(lrl2.predict(Xtest), Ytest))
    graph = [l2, l2test]
    color = ["black", "gray"]
    label = ["L2", "L2test"]
    plt.figure(figsize=(20, 5))
    for i in range(len(graph)):
        plt.plot(np.arange(1, 201, 10), graph[i], color[i], label=label[i])
    plt.legend(loc=4)
    plt.xticks(np.arange(1, 201, 10))
    plt.show()
    # 属性.n_iter_表示本次求解中真正实现的迭代次数
    lr = LR(penalty="l2", solver="liblinear", C=0.9, max_iter=300).fit(Xtrain, Ytrain)
    print(lr.n_iter_)  # [24]


def multi_class_lr():
    '''
    sklearn提供了多种可以使用逻辑回归处理多分类问题的选项。
    可以把某种分类类型都看作1，其余的分类类型都为0值，这种方法被称为"一对多"(One-vs-rest)，简称OvR，在sklearn中表示为“ovr"；
    又或者，可以把好几个分类类型划为1，剩下的几个分类类型划为0值，这是一种”多对多“(Many-vs-Many)的方法，简称MvM，在sklearn中表示为"Multinominal"。
    :return:
    '''
    ## 加载鸢尾花三分类数据集
    iris = load_iris()
    ## 鸢尾花数据集上，multinomial和ovr的区别怎么样：
    # 'ovr': 表示分类问题是二分类，或让模型使用"一对多"的形式来处理多分类问题。
    # 'multinomial'：表示处理多分类问题，这种输入在参数solver是'liblinear'时不可用。
    for multi_class in ["multinomial", "ovr"]:
        clf = LR(solver="sag", max_iter=100, random_state=42, multi_class=multi_class).fit(iris.data,
                                                                                           iris.target)  # "sag"是随机平均梯度下降
        # 打印两种multi_class模式下的训练分数
        print("training score is %.3f (%s)." % (clf.score(iris.data, iris.target), multi_class))  # 0.987  0.960


if __name__ == "__main__":
    # 逻辑回归
    # lr()
    # 逻辑回归中的特征工程
    # feature_engining_in_lr()
    # max_iter的学习曲线
    # max_iter_parameter()
    # 逻辑回归的多分类
    multi_class_lr()
