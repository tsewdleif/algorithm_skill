#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/6/29 0029 下午 12:35
# @Author : West Field
# @File : algo.py

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics import calinski_harabaz_score  # 卡林斯基-哈拉巴斯指数
from sklearn.metrics import davies_bouldin_score  # 戴维斯-布尔丁指数
from sklearn.metrics.cluster import contingency_matrix  # 权变矩阵
import matplotlib.cm as cm
import numpy as np


def kmeans_cluster():
    '''
    聚类算法其目的是将数据划分成有意义或有用的组（或簇）。
    无监督的算法在训练的时候只需要特征矩阵X，不需要标签。PCA降维算法就是无监督学习中的一种，聚类算法也是无监督学习的代表算法之一。
    :return:
    '''
    ## 自己创建数据集
    X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)
    # fig是画布，ax1是子图对象
    fig, ax1 = plt.subplots(1)  # 生成一个子图
    ax1.scatter(X[:, 0], X[:, 1],
                marker='o',  # 点的形状
                s=8)  # 点的大小
    plt.show()
    ## 使用Kmeans进行聚类
    # 先聚成3类观察效果
    n_clusters = 3
    cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    print(cluster.labels_)  # 查看聚类完成之后的每个样本的类别
    # 学习数据X并对X的类进行预测
    y_pred = cluster.fit_predict(X)
    print(y_pred)
    # 查看类的质心
    centroid = cluster.cluster_centers_
    print(centroid)
    # 查看总的距离平方和
    print(cluster.inertia_)
    # 对聚类结果画图
    color = ["red", "pink", "orange", "gray"]
    fig, ax1 = plt.subplots(1)
    for i in range(n_clusters): ax1.scatter(X[y_pred == i, 0], X[y_pred == i, 1], marker='o', s=8, c=color[i])
    ax1.scatter(centroid[:, 0], centroid[:, 1], marker="x", s=15, c="black")
    plt.show()
    ## 聚类算法的模型评估指标
    # inertia_不适合作为评价指标，它只是样本到质心的距离总和，会随着质心个数增加递减。
    # 当真实标签已知，评估指标：互信息分、V-measure、调整兰德系数
    # 当真实标签未知，评估指标：轮廓系数(簇内差异小，簇外差异大，得到的系数值越接近于1越好，但它在凸型的类上表现会虚高。)
    print(silhouette_score(X, y_pred))  # 聚类结果的轮廓系数
    print(silhouette_score(X, KMeans(n_clusters=4, random_state=0).fit_predict(X)))  # 发现分为4类的轮廓系数最大
    print(silhouette_samples(X, y_pred))  # 每一个样本对应的轮廓系数
    # 除了轮廓系数是最常用的，我们还有卡林斯基-哈拉巴斯指数（Calinski-Harabaz   Index，简称CHI，也被称为方差比标准），戴维斯-布尔丁指数（Davies-Bouldin）以及权变矩阵（Contingency Matrix）可以使用。
    # 卡林斯基-哈拉巴斯指数越大越好，在凸型的数据上的聚类也会表现虚高。但是比起轮廓系数，它有一个巨大的优点，就是计算非常快速。
    print(calinski_harabaz_score(X, y_pred))


def choose_n_cluster():
    '''
    基于轮廓系数来选择n_clusters
    :return:
    '''
    # 创建数据库
    X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)
    # 绘制轮廓系数分布图和聚类后的数据分布图来选择我们的最佳n_clusters
    for n_clusters in [2, 3, 4, 5, 6, 7]:
        # 创建子图，1行2个子图
        fig, (ax1, ax2) = plt.subplots(1, 2)
        # 设置整个画布大小
        fig.set_size_inches(18, 7)
        # 设置x坐标的范围，由于轮廓系数在-1到1之间，而且一般是大于0的，所以这里设置为-0.1到1
        ax1.set_xlim([-0.1, 1])
        # 设置y的坐标范围，加50是为了让y坐标的条形图之间有间隔
        ax1.set_ylim([0, X.shape[0] + 50])
        # 聚类建模
        clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X)
        cluster_labels = clusterer.labels_
        # 整体样本的平均轮廓系数
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
        # 每个样本的轮廓系数
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        # 设置y轴的下限
        y_lower = 10
        for i in range(n_clusters):
            # 当分簇为n_clusters时，聚类结果中第i个簇的每个样本的轮廓系数
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            # 对第i个簇中的轮廓系数排序
            ith_cluster_silhouette_values.sort()
            # 第i簇的样本个数
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            # 第i个簇样本的y值上限
            y_upper = y_lower + size_cluster_i
            # nipy_spectral输入任意小数来代表一个颜色，每个簇的颜色设置为不同颜色
            color = cm.nipy_spectral(float(i) / n_clusters)
            # 填充子图1的内容，fill_betweenx的范围是在纵坐标上
            ax1.fill_betweenx(np.arange(y_lower, y_upper), ith_cluster_silhouette_values, facecolor=color,
                              alpha=0.7)  # alpha是透明度
            # 为每个簇的轮廓系数写上簇的编号，-0.05是编号的横坐标，然后是纵坐标，然后是簇的编号
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            # 为下一个簇计算新的y轴上的初始值
            y_lower = y_upper + 10
        # 给子图1加上标题、横纵坐标的名称
        ax1.set_title("\nThe silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        # 把整个数据集上的轮廓系数的均值以虚线的形式放入图中，大于均值的簇分类结果较好
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        # 让y轴不显示刻度
        ax1.set_yticks([])
        # 设置x轴刻度
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        # 开始画第二个子图
        # 一次性获取多个小数来获取多个颜色
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        # 给子图2画散点图。x、y坐标，点的形状，点的大小，点的颜色。
        ax2.scatter(X[:, 0], X[:, 1], marker='o', s=8, c=colors)
        # 簇的质心
        centers = clusterer.cluster_centers_
        # 把生成的质心放到图像中去
        ax2.scatter(centers[:, 0], centers[:, 1], marker='x', c="red", alpha=1, s=200)
        # 子图2的标题，x、y轴的标题
        ax2.set_title("\nThe visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        # 为整张大图设置标题
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
        plt.show()
        # 结果发现分2类时的轮廓系数最大，其次是分4类的轮廓系数。
        # 这种情况下就需要根据实际的业务需求或者实际的资源条件来分2类或者4类。
        # 例如，发广告传单，如果人力资源有限可以将客户分为两类进行精准营销方案设计，如果人力资源丰富，可以分为四类客户进行精准营销方案设计。
        # 或者先分为两类进行精准营销，等有了盈利，再分为四类进行精准营销也是可以的。


def stop_iter():
    '''
    参数max_iter、tol控制迭代次数：
    默认当质心不再移动，Kmeans算法停止迭代停止。
    但在完全收敛之前，我们也可以使用max_iter(最大迭代次数)或tol(两次迭代间Inertia下降的量)，这两个参数来让迭代提前停下来。
    :return:
    '''
    # 创建数据库
    X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)
    # 初始化质心的方式设定为random
    random1 = KMeans(n_clusters=10, init="random", max_iter=5, random_state=420).fit(X)
    y_pred_max10 = random1.labels_
    # 通过轮廓系数看聚类的效果
    print(silhouette_score(X, y_pred_max10))
    # 最大迭代次数设置为20
    random2 = KMeans(n_clusters=10, init="random", max_iter=10, random_state=420).fit(X)
    y_pred_max20 = random2.labels_
    print(silhouette_score(X, y_pred_max20))


if __name__ == "__main__":
    # 聚类算法
    # kmeans_cluster()
    # 基于轮廓系数来选择n_clusters
    # choose_n_cluster()
    # 让迭代停下来
    stop_iter()
