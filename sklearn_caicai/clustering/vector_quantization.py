#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/6/30 0030 下午 16:35
# @Author : West Field
# @File : vector_quantization.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin  # 对两个序列中的点进行距离匹配
from sklearn.datasets import load_sample_image  # 导入图片所用的类
from sklearn.utils import shuffle
import pandas as pd


def vq():
    '''
    K-Means聚类最重要的应用之一是非结构数据(图像，声音)上的矢量量化（VQ）。
    特征选择的降维是直接选取对模型贡献最大的特征，PCA的降维是聚合信息，而矢量量化的降维是在同等样本量上压缩信息的大小，即不改变特征的数目也不改变样本的数目，只改变在这些特征下的样本上的信息量。
    :return:
    '''
    ## 导入数据，探索数据
    china = load_sample_image("china.jpg")
    print(china)
    print(type(china), china.dtype, china.shape)  # <class 'numpy.ndarray'> uint8 (427, 640, 3)
    # 包含多少种非重复颜色，一个唯一的(R, G, B)值算一种颜色
    newimage = china.reshape((427 * 640, 3))  # 降成二维数据
    print(newimage.shape)  # (273280, 3)
    # 该图片中所有独一无二的颜色就是该图片所带有的信息量
    print(pd.DataFrame(newimage).drop_duplicates().shape)  # (96615, 3)
    # 图像可视化
    plt.figure(figsize=(15, 15))
    plt.imshow(china)  # 参数要是三维数组x形成的图片
    plt.show()
    # 9万多种颜色太多了，尝试将颜色压缩到64中，还不损耗图像的质量。使用64个簇的质心代替这9万个颜色。
    ## 数据预处理
    # 归一化特征
    china = np.array(china, dtype=np.float64) / china.max()
    w, h, d = tuple(china.shape)
    # 将三维数据转化为二维，KMeans接收的是二维数据
    assert d == 3  # 如果d不为3，则报错
    image_array = np.reshape(china, (w * h, d))
    print(image_array)
    ## 对数据进行K-Means的矢量量化
    n_clusters = 64
    # 由于数据量太大，先随机选取1000个数据来找出质心，然后把其它样本根据这些质心聚类
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_array_sample)
    print(kmeans.cluster_centers_.shape)  # (64, 3) 质心
    # 找出质心之后，按照已存在的质心对所有数据进行聚类
    labels = kmeans.predict(image_array)  # lables中是所有样本的簇的质心的索引
    print(labels.shape)  # (273280,)
    # 用质心替换掉所有的样本，原理是：同一簇的点是相似的
    image_kmeans = image_array.copy()  # image_kmeans中是27万个样本点、9万多种不同的颜色
    # 遍历所有样本
    for i in range(w * h):
        image_kmeans[i] = kmeans.cluster_centers_[labels[i]]
    # 查看生成的新图片信息
    print(image_kmeans.shape)  # (273280, 3)
    # 查看里边还有多少种独一无二的颜色
    print(pd.DataFrame(image_kmeans).drop_duplicates().shape)  # (64, 3)
    # 将图像从二维恢复到三维，但它剩的颜色只有64种了
    image_kmeans = image_kmeans.reshape(w, h, d)
    print(image_kmeans.shape)  # (427, 640, 3)
    ## 对数据进行随机的矢量量化，即随机的从样本中选取64个样本点作为质心
    # 随机抽取64个样本作为质心
    centroid_random = shuffle(image_array, random_state=0)[:n_clusters]
    # 计算image_array中每个样本到centroid_random中每个质心的距离，并返回与image_array相同形状的，在centroid_random中最近的质心的索引
    labels_random = pairwise_distances_argmin(centroid_random, image_array, axis=0)
    print(labels_random.shape)  # (273280,)
    print(labels_random)
    # 使用随机质心来替换所有样本
    image_random = image_array.copy()
    for i in range(w * h):
        image_random[i] = centroid_random[labels_random[i]]
    # 将图像从二维恢复到三维，这里的64种颜色是随机选出来的，上边使用KMeans量化出的64种颜色是通过聚类聚出来的
    image_random = image_random.reshape(w, h, d)
    print(image_random.shape)  # (427, 640, 3)
    ## 将原图、按KMeans矢量量化、随机矢量量化的图像分别绘制出来
    # 原图，9万多种颜色的图
    plt.figure(figsize=(10, 10))
    plt.axis('off')  # 不要显示坐标轴
    plt.title('Original image (96,615 colors)')
    plt.imshow(china)
    # KMeans量化之后的图像，64种颜色。会发现成功的使用64种颜色展示了一个和原图差不多的图像。
    # 矢量量化后的图像所占的空间大小就要小很多很多了，原图9万多种颜色，矢量量化的图像只有64种颜色，实现了对数据进行压缩。
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.title('Quantized image (64 colors, K-Means)')
    plt.imshow(image_kmeans)
    # 随机量化之后的图像。会发现颜色和原图相差有些大，损失较多，不如KMeans矢量量化的效果好。
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.title('Quantized image (64 colors, Random)')
    plt.imshow(image_random)
    plt.show()


if __name__ == "__main__":
    # 矢量量化
    vq()
