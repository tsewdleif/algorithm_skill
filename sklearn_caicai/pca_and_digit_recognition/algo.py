#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/6/24 0024 上午 0:21
# @Author : West Field
# @File : algo.py

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import load_digits

'''
特征选择和降维算法的区别：
特征选择是从已存在的特征中选取携带信息最多的，并未修改原先特征。
而降维算法如PCA，是将已存在的特征进行压缩，降维完毕后的特征不是原本的特征矩阵中的任何一个特征，而是通过某
些方式组合起来的新特征。以PCA为代表的降维算法因此是特征创造的一种。
'''
'''
PCA和SVD是两种不同的降维算法，它们实现降维的过程相同，只是两种算法中矩阵分解的方法不同，信息量的衡量指标不同罢了。
PCA使用方差作为信息量的衡量指标，SVD使用奇异值来衡量特征上的信息量的指标。
因此，降维算法的计算量很大，运行比较缓慢，但无论如何，它们的功能无可替代，它们依然是机器学习领域的宠儿。
'''


def pca_decomposition():
    ## 加载鸢尾花数据
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(X.shape, y.shape)
    ## PCA降维，从4维降到2维
    pca = PCA(n_components=2)
    X_dr = pca.fit_transform(X)
    print(X_dr)
    # 将降维后的数据可视化
    print(X_dr[y == 0, 0])  # 标签为0的第一列数据
    plt.figure()
    plt.scatter(X_dr[y == 0, 0], X_dr[y == 0, 1], c="red", label=iris.target_names[0])
    plt.scatter(X_dr[y == 1, 0], X_dr[y == 1, 1], c="black", label=iris.target_names[1])
    plt.scatter(X_dr[y == 2, 0], X_dr[y == 2, 1], c="orange", label=iris.target_names[2])
    plt.legend()
    plt.show()
    # 降维后的特征的重要性，信息都集中在靠前的特征上
    print(pca.explained_variance_)
    # 可解释方差贡献率，查看降维后每个新特征带有的信息量占原始数据总信息量的百分比
    print(pca.explained_variance_ratio_, pca.explained_variance_ratio_.sum())
    # 画总可解释性方差贡献率曲线，横坐标是PCA降维之后的特征维数，横坐标为降维后的所有特征可解释性方差贡献率总和
    pca_line = PCA().fit(X)  # PCA()默认n_components为原数据集特征个数，只是将数据投影到新的特征空间，未改变特征个数
    print(pca_line.explained_variance_ratio_, np.cumsum(pca_line.explained_variance_ratio_))
    plt.plot([1, 2, 3, 4], np.cumsum(pca_line.explained_variance_ratio_))
    plt.xticks([1, 2, 3, 4])
    plt.xlabel("number of components after dimension reduction")
    plt.ylabel("cumulative explained variance ratio")
    plt.show()
    ## 使用最大似然估计自动选择n_components超参数
    pca_mle = PCA(n_components="mle").fit(X)
    X_mle = pca_mle.transform(X)
    print(X_mle)
    print(pca_mle.explained_variance_ratio_.sum())
    ## 按信息量占比选择n_components超参数
    pca_f = PCA(n_components=0.97, svd_solver="full")  # 选择信息量占原始数据集达到97%时的特征作为PCA降维后的特征
    X_f = pca_f.fit_transform(X)
    print(X_f)
    print(pca_f.explained_variance_ratio_,
          pca_f.explained_variance_ratio_.sum())  # [0.92461872 0.05306648] 0.977685206318795


def inverse_transform_function():
    '''
    重要接口inverse_transform：
    降维后再通过inverse_transform转换回原维度的数据画出的图像和原数据画的图像大致相似，但原数据的图像明显更加清晰。
    这说明，inverse_transform并没有实现数据的完全逆转。
    这是因为，在降维的时候，部分信息已经被舍弃了，X_dr中往往不会包含原数据100%的信息，所以在逆转的时
    候，即便维度升高，原数据中已经被舍弃的信息也不可能再回来了。所以，降维不是完全可逆的。
    :return:
    '''
    # 加载人脸数据，七个人，一千多张图片
    faces = fetch_lfw_people(min_faces_per_person=60)  # 提取的数据集将仅保留具有以下特征的人的照片：最少有60张不同的图片。
    print(faces.images.shape, faces.data.shape)
    X = faces.data
    # 建模降维，从2914维降到150维
    pca = PCA(150)
    X_dr = pca.fit_transform(X)
    print(X_dr.shape)
    X_inverse = pca.inverse_transform(X_dr)
    print(X_inverse.shape)
    # 将特征矩阵X和X_inverse可视化
    fig, ax = plt.subplots(2, 10, figsize=(10, 2.5), subplot_kw={"xticks": [], "yticks": []})  # 不要显示坐标轴
    # 填充子图
    for i in range(10):
        ax[0, i].imshow(faces.images[i, :, :], cmap="binary_r")  # 选择色彩的模式
        ax[1, i].imshow(X_inverse[i].reshape(62, 47), cmap="binary_r")
    plt.show()


def noise_filter():
    '''
    inverse_transform能够在不恢复原始数据的情况下，将降维后的数据返回到原本的高维空间，即是说能够实现保证维度、
    但去掉方差很小特征所带的信息。利用inverse_transform的这个性质，我们能够实现噪音过滤。
    :return:
    '''
    # 导入手写数字识别数据集
    digits = load_digits()
    print(digits.data.shape)

    # 定义画图函数
    def plot_digits(data):
        fig, axes = plt.subplots(4, 10, figsize=(10, 4), subplot_kw={"xticks": [], "yticks": []})
        for i, ax in enumerate(axes.flat):
            ax.imshow(data[i].reshape(8, 8), cmap="binary")
        plt.show()

    plot_digits(digits.data)
    # 为数据加上噪音
    np.random.RandomState(42)
    noisy = np.random.normal(digits.data, 2)  # 在指定的数据集中，随机抽取服从正态分布的数据，2是抽取出来的正太分布的方差
    plot_digits(noisy)
    # 降维
    pca = PCA(0.9).fit(noisy)  # 保留90%的信息量
    X_dr = pca.transform(noisy)
    print(X_dr.shape)
    # 逆转降维结果，实现降噪
    without_noise = pca.inverse_transform(X_dr)
    plot_digits(without_noise)


if __name__ == "__main__":
    # PCA降维
    # pca_decomposition()
    # 重要接口inverse_transform
    # inverse_transform_function()
    # PCA做数据噪音过滤
    noise_filter()
