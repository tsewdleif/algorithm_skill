#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/6/26 0026 上午 0:19
# @Author : West Field
# @File : digit_recognition.py

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
import time


def pca_digit_recognition():
    '''
    数据集结构为(42000, 784)，用KNN跑一次半小时，得到准确率在96.6%上下，用随机森林跑一次12秒，准确率在93.8%，
    虽然KNN效果好，但由于数据量太大，KNN计算太缓慢，所以我们不得不选用随机森林。
    我们使用了各种技术对手写数据集进行特征选择，最后使用嵌入法SelectFromModel选出了324个特征，将随机森林的效果也调到了96%以上。
    但是，因为数据量依然巨大，还是有300多个特征。现在，我们就来试着用PCA处理一下这个数据，看看效果如何。
    :return:
    '''
    # 导入手写数字识别数据集
    data = pd.read_csv(
        r"D:\python\PythonProject\algorithm_skill\sklearn_caicai\data_preprocessing_and_feature_engineering\digit recognizor.csv")
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    print(X.shape)
    # 画累计方差贡献率曲线，找最佳降维后维度的范围
    # pca_line = PCA().fit(X)
    # plt.figure(figsize=[20, 5])
    # plt.plot(np.cumsum(pca_line.explained_variance_ratio_))
    # plt.xlabel("number of components after dimension reduction")
    # plt.ylabel("cumulative explained variance ratio")
    # plt.show()
    # # components取100时曲线趋于平缓，继续缩小最佳维度的范围
    # score = []
    # for i in range(1, 101, 10):
    #     X_dr = PCA(i).fit_transform(X)
    #     once = cross_val_score(RFC(n_estimators=10, random_state=0), X_dr, y, cv=5).mean()
    #     score.append(once)
    # plt.figure(figsize=[20, 5])
    # plt.plot(range(1, 101, 10), score)
    # plt.show()
    # 细化学习曲线，找出降维后的最佳维度
    # score = []
    # for i in range(10, 25):
    #     X_dr = PCA(i).fit_transform(X)
    #     once = cross_val_score(RFC(n_estimators=10, random_state=0), X_dr, y, cv=5).mean()
    #     score.append(once)
    # plt.figure(figsize=[20, 5])
    # plt.plot(range(10, 25), score)
    # plt.show()
    # 导入找出的最佳维度进行降维，查看模型效果
    X_dr = PCA(21).fit_transform(X)
    # print(cross_val_score(RFC(n_estimators=100, random_state=0), X_dr, y, cv=5).mean()) # 0.9438809523809523
    # 跑出了94.49%的水平，但还是没有我们使用嵌入法特征选择过后的96%高，有没有什么办法能够  提高模型的表现呢？
    # 我们知道KNN的效果比随机森林更好，只是之前未特征降维之前维度高，KNN计算量太大，现在看将为之后的KNN运行时间和准确率如何
    # print(cross_val_score(KNN(), X_dr, y, cv=5).mean())  # 0.9676666666666666
    # KNN的k值学习曲线
    # score = []
    # for i in range(10):
    #     # X_dr = PCA(21).fit_transform(X)
    #     once = cross_val_score(KNN(i + 1), X_dr, y, cv=5).mean()
    #     score.append(once)
    # plt.figure(figsize=[20, 5])
    # plt.plot(range(10), score)
    # plt.show()
    # 定下超参数后，模型效果如何，模型运行时间如何？
    t1 = time.time()
    print(cross_val_score(KNN(3), X_dr, y, cv=5).mean())  # 0.9682142857142857
    print(time.time() - t1)  # 42.757675886154175
    # 可以发现，原本785列的特征被我们缩减到23列之后，用KNN跑出了目前位置这个数据集上最好的结果。
    # PCA为我们提供了无限的可能，终于不用再因为数据量太庞大而被迫选择更加复杂的模型了！


if __name__ == "__main__":
    # PCA对手写数字识别的降维
    pca_digit_recognition()
