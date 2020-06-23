#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/6/22 0022 下午 22:00
# @Author : West Field
# @File : feature_selection.py

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score
import time
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE

'''
数据预处理完成之后，就开始进行特征工程了。
特征工程有三种：
特征提取(从文字，图像，声音等其他非结构化数据中提取新信息作为特征)；
特征创造(把现有特征进行组合，或互相计算，得到新的特征)；
特征选择(从所有的特征中，选择出有意义，对模型有帮助的特征，以避免必须将所有特征都导入模型去训练的情况)。
有四种方法可以用来选择特征：过滤法，嵌入法，包装法，和降维算法。
'''


def variance_filter():
    '''
    1.1、方差过滤法
    现实中，我们只会使用阈值为0或者阈值很小的方差过滤，来为我们优先消除一些明显用不到的特征，然后我们会选择更优的特征选择方法继续削减特征数量。
    :return:
    '''
    ## 导入手写数字识别数据集
    data = pd.read_csv("./digit recognizor.csv")
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    print(X.shape)
    ## 方差过滤：根据特征本身的方差来筛选特征
    # 删除方差为0的特征
    selector = VarianceThreshold()  # 实例化，不填参数默认方差为0
    X_var0 = selector.fit_transform(X)  # 获取删除不合格特征之后的新特征矩阵
    print(X_var0.shape)
    # 删除小于所有特征中位数的方差的特征，删除后剩余一半特征
    X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)
    print(X_fsvar.shape)
    # 当特征是二分类时，删除其中某种分类占到80%以上的特征
    X_bvar = VarianceThreshold(0.8 * (1 - 0.8)).fit_transform(X)
    print(X_bvar.shape)
    ## 方差过滤对模型的影响：效果对比(模型的准确率；运行时间)，这里对比随机森林和KNN在特征方差过滤前后的效果
    # KNN在特征方差过滤前后对比
    t = time.time()
    knn_score = cross_val_score(KNN(), X, y, cv=5).mean()
    print(knn_score, time.time() - t)  # 0.9658569700264943, 特征方差过滤前，大约需要半小时时间
    t = time.time()
    knn_score = cross_val_score(KNN(), X_fsvar, y, cv=5).mean()
    print(knn_score, time.time() - t)  # 0.9659997478180573, 特征方差过滤后，大约需要20分钟时间
    # RFC在特征方差过滤前后对比
    t = time.time()
    rfc_score = cross_val_score(RFC(n_estimators=10, random_state=0), X, y, cv=5).mean()
    print(rfc_score, time.time() - t)  # 0.9373571428571429, 特征方差过滤前，大约需要17.8秒
    t = time.time()
    rfc_score = cross_val_score(RFC(n_estimators=10, random_state=0), X_fsvar, y, cv=5).mean()
    print(rfc_score, time.time() - t)  # 0.9390476190476191, 特征方差过滤后，大约需要18.7秒


def correlation_filter():
    '''
    1.2、相关性过滤
    我们希望选出与标签相关且有意义的特征，有三种常用的方法来评判特征与标签之间的相关性：卡方，F检验，互信息。
    :return:
    '''
    ## 导入手写数字识别数据集
    data = pd.read_csv("./digit recognizor.csv")
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    print(X.shape)
    ## 卡方过滤：计算每个非负特征和标签之间的卡方统计量，并依照卡方统计量由高到低为特征排名。
    # 删除小于所有特征中位数的方差的特征，删除后剩余一半特征
    X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)
    print(X_fsvar.shape)
    # 假设在这里需要300个特征
    X_fschi = SelectKBest(chi2, k=300).fit_transform(X_fsvar, y)
    print(X_fschi.shape)
    print(cross_val_score(RFC(n_estimators=10, random_state=0), X_fschi, y, cv=5).mean())
    # 学习超参数K
    # score = []
    # for i in range(390, 200, -10):
    #     X_fschi = SelectKBest(chi2, k=i).fit_transform(X_fsvar, y)
    #     once = cross_val_score(RFC(n_estimators=10, random_state=0), X_fschi, y, cv=5).mean()
    #     score.append(once)
    # plt.plot(range(390, 200, -10), score)
    # plt.show() # 图像一直上升
    # 根据卡方值和P值确定K值
    chivalue, pvalues_chi = chi2(X_fsvar, y)
    print(chivalue, pvalues_chi)
    # k取多少？我们想要消除所有p值大于设定值，比如0.05或0.01的特征：
    k = chivalue.shape[0] - (pvalues_chi > 0.05).sum()
    print(k)  # 392
    ## F检验，又称ANOVA，方差齐性检验，是用来捕捉每个特征与标签之间的线性关系的过滤方法。
    F, pvalues_f = f_classif(X_fsvar, y)
    print(F, pvalues_f)
    k = F.shape[0] - (pvalues_f > 0.05).sum()
    print(k)  # 392
    ## 互信息法，它是用来捕捉每个特征与标签之间的任意关系（包括线性和非线性关系）的过滤方法。
    # 它返回“每个特征与目标之间的互信息量的估计”，这个估计量在[0,1]之间取值，为0则表示两个变量独立，为1则表示两个变量完全相关。
    result = MIC(X_fsvar, y)
    k = result.shape[0] - sum(result <= 0)
    print(k)  # 392


def embedded():
    '''
    2、嵌入法
    嵌入法是一种让算法自己决定使用哪些特征的方法，即特征选择和算法训练同时进行。在使用嵌入法时，我们先使
    用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据权值系数从大到小选择特征。这些权值系
    数往往代表了特征对于模型的某种贡献或某种重要性，比如决策树和树的集成模型中的feature_importances_属
    性，可以列出各个特征对树的建立的贡献，我们就可以基于这种贡献的评估，找出对模型建立最有用的特征。
    :return:
    '''
    ## 导入手写数字识别数据集
    data = pd.read_csv("./digit recognizor.csv")
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    ## 嵌入法选择特征
    RFC_ = RFC(n_estimators=10, random_state=0)
    # 在这里我只想取出来有限的特征。0.005这个阈值对于有780个特征的数据来说，是非常高的阈值，因为平均每个特征只能够分到大约0.001的feature_importances_
    X_embedded = SelectFromModel(RFC_, threshold=0.005).fit_transform(X, y)
    print(X_embedded.shape)  # (42000, 47)
    # 画学习曲线来找最佳阈值
    print(RFC_.fit(X, y).feature_importances_)
    threshold = np.linspace(0, (RFC_.fit(X, y).feature_importances_).max(), 20)
    # score = []
    # for i in threshold:
    #     X_embedded = SelectFromModel(RFC_, threshold=i).fit_transform(X, y)
    #     once = cross_val_score(RFC_, X_embedded, y, cv=5).mean()
    #     score.append(once)
    # plt.plot(threshold, score)
    # plt.show()
    # 观察上图，阈值在0.00134之前还可以，再细分找阈值
    # score2 = []
    # for i in np.linspace(0,0.00134,20):
    #     X_embedded = SelectFromModel(RFC_,threshold=i).fit_transform(X,y)
    #     once = cross_val_score(RFC_,X_embedded,y,cv=5).mean()
    #     score2.append(once)
    # plt.figure(figsize=[20,5])
    # plt.plot(np.linspace(0,0.00134,20),score2)
    # plt.xticks(np.linspace(0,0.00134,20))
    # plt.show()
    # 使用嵌入法选择的特征训练出最优模型
    X_embedded = SelectFromModel(RFC_, threshold=0.000071).fit_transform(X, y)
    print(X_embedded.shape)
    print(cross_val_score(RFC_, X_embedded, y, cv=5).mean())  # 0.9388809523809524
    # 我们可能已经找到了现有模型下的最佳结果，如果我们调整一下随机森林的参数会更优
    print(cross_val_score(RFC(n_estimators=100, random_state=0), X_embedded, y, cv=5).mean())  # 0.964642857142857


def wrapper():
    '''
    3、包装法
    包装法也是一个特征选择和算法训练同时进行的方法，与嵌入法十分相似，它也是依赖于算法自身的选择，比如
    coef_属性或feature_importances_属性来完成特征选择。但不同的是，我们往往使用一个目标函数作为黑盒来帮
    助我们选取特征，而不是自己输入某个评估指标或统计量的阈值。
    最典型的包装法目标函数是递归特征消除法（Recursive feature elimination, 简写为RFE）。它是一种贪婪的优化算法，
    旨在找到性能最佳的特征子集。 它反复创建模型，并在每次迭代时保留最佳特征或剔除最差特征，下一次迭代时，
    它会使用上一次建模中没有被选中的特征来构建下一个模型，直到所有特征都耗尽为止。 然后，它根据自己保留或
    剔除特征的顺序来对特征进行排名，最终选出一个最佳子集。
    :return:
    '''
    ## 导入手写数字识别数据集
    data = pd.read_csv("./digit recognizor.csv")
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    # 包装法
    RFC_ = RFC(n_estimators=10, random_state=0)
    # n_features_to_select是想要选择的特征个数，step表示每次迭代中希望移除的特征个数。
    selector = RFE(RFC_, n_features_to_select=340, step=50).fit(X, y)
    print(selector.support_)  # 所有的特征的是否最后被选中的布尔矩阵
    print(selector.ranking_)  # 特征综合重要性的排名
    # 使用包装法选择的特征集
    X_wrapper = selector.transform(X)
    print(cross_val_score(RFC_, X_wrapper, y, cv=5).mean())  # 0.9379761904761905
    # 对包装法画学习曲线选择n_features_to_select
    score = []
    for i in range(1, 751, 50):
        X_wrapper = RFE(RFC_, n_features_to_select=i, step=50).fit_transform(X, y)
        once = cross_val_score(RFC_, X_wrapper, y, cv=5).mean()
        score.append(once)
    plt.figure(figsize=[20, 5])
    plt.plot(range(1, 751, 50), score)
    plt.xticks(range(1, 751, 50))
    plt.show()
    # 能够看出，在包装法下面，应用50个特征时，模型的表现就已经达到了90%以上，比嵌入法和过滤法都高效很多。


if __name__ == "__main__":
    # 方差过滤
    # variance_filter()
    # 相关性过滤
    # correlation_filter()
    # 嵌入法
    # embedded()
    # 包装法
    wrapper()
