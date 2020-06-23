#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/6/20 0020 下午 21:18
# @Author : West Field
# @File : data_preprocessing.py

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder  # 标签专用，能够将分类转换成分类数值
from sklearn.preprocessing import OrdinalEncoder  # 特征专用，能够将分类特征转换为分类数值
from sklearn.preprocessing import OneHotEncoder  # 独特编码，创建哑变量
from sklearn.preprocessing import Binarizer  # 将连续性数据二值化
from sklearn.preprocessing import KBinsDiscretizer  # 将连续性数据分段

'''
数据挖掘的五大流程：获取数据；数据预处理；特征工程；建模，测试模型并预测出结果；上线，验证模型效果。
数据预处理：从数据中检测，纠正或删除损坏，不准确或不适用于模型的记录的过程，如数据类型不同、数据偏态等。
特征工程：特征工程是将原始数据转换为更能代表预测模型的潜在问题的特征的过程。
'''


def nondimensionalization():
    '''
    1、数据无量纲化：
    将不同规格的数据转换到同一规格，或不同分布的数据转换到某个特定分布的需求，这种需求统称为将数据“无量纲化”。
    线性的无量纲化包括中心化（Zero-centered或者Meansubtraction）处理和缩放处理（Scale）。
    :return:
    '''
    ## MinMaxScaler归一化
    data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    data = pd.DataFrame(data)
    scaler = MinMaxScaler()  # 实例化
    scaler = scaler.fit(data)  # fit，在这里本质是生成min(x)和max(x)
    result = scaler.transform(data)  # 通过接口导出结果
    print(result)
    result_ = scaler.fit_transform(data)  # 训练和导出结果一步达成
    print(result_)
    print(scaler.inverse_transform(result))  # 将归一化后的结果逆转
    # 使用MinMaxScaler的参数feature_range实现将数据归一化到[0,1]以外的范围中
    data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    scaler = MinMaxScaler(feature_range=[5, 10])  # 依然实例化
    result = scaler.fit_transform(data)  # fit_transform一步导出结果
    print(result)
    ## StandardScaler标准化
    data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    scaler = StandardScaler()  # 实例化
    scaler.fit(data)  # fit，本质是生成均值和方差
    print(scaler.mean_, scaler.var_)  # 查看均值的属性mean_、方差的属性var_
    x_std = scaler.transform(data)  # 通过接口导出结果
    print(x_std.mean(), x_std.std())  # 导出的结果是一个数组，用mean()查看均值、用std()查看方差
    x_std = scaler.fit_transform(data)  # 使用fit_transform(data)一步达成结果
    print(x_std)
    print(scaler.inverse_transform(x_std))  # 使用inverse_transform逆转标准化


def missing_value():
    '''
    2、缺失值填充
    :return:
    '''
    # 导入缺失的数据集
    data = pd.read_csv("./Narrativedata.csv", index_col=0)
    print(data.head())
    print(data.info())
    # 填补年龄
    Age = data.loc[:, "Age"].values.reshape(-1, 1)  # sklearn当中特征矩阵必须是二维
    print(Age[:20])
    imp_mean = SimpleImputer()  # 实例化，默认均值填补
    imp_median = SimpleImputer(strategy="median")  # 用中位数填补
    imp_0 = SimpleImputer(strategy="constant", fill_value=0)  # 用0填补
    imp_mean = imp_mean.fit_transform(Age)  # fit_transform一步完成调取结果
    imp_median = imp_median.fit_transform(Age)
    imp_0 = imp_0.fit_transform(Age)
    print(imp_mean[:20])
    print(imp_median[:20])
    print(imp_0[:20])
    # 在这里我们使用中位数填补Age
    data.loc[:, "Age"] = imp_median
    print(data.info())
    # 使用众数填补Embarked
    Embarked = data.loc[:, "Embarked"].values.reshape(-1, 1)
    imp_mode = SimpleImputer(strategy="most_frequent")
    data.loc[:, "Embarked"] = imp_mode.fit_transform(Embarked)
    print(data.info())
    # 返回无缺失值的数据
    return data


def coding_and_dummy_variables():
    '''
    3、处理分类型特征：编码与哑变量
    机器学习中大多数算法只能够处理数值型数据，如LR、SVM、KNN等，决策树和朴素贝叶斯可以处理文字数据，但是它们在sklearn中不可以。
    为了让数据适应算法和库，我们必须将数据进行编码，即是说，将文字型数据转换为数值型。
    三种不同性质的分类数据：名义变量(特征取值之间相互独立，如舱门类型)；有序变量(特征取值之间有序，如学历)；有距变量(如体重分段)。
    类别OrdinalEncoder可以用来处理有序变量，但对于名义变量，我们只有使用哑变量的方式来处理，才能够尽量向算法传达最准确的信息。
    :return:
    '''
    ## 标签编码
    data = missing_value()
    print(data.head())
    print(data.info())
    y = data.iloc[:, -1]  # 要输入的是标签，不是特征矩阵，所以允许一维
    le = LabelEncoder()  # 实例化
    le = le.fit(y)  # 导入数据
    label = le.transform(y)  # transform接口调取结果
    print(le.classes_)  # classes_查看标签中究竟有多少类别
    print(label)  # 查看获取的结果label
    print(le.fit_transform(y))  # 也可以直接fit_transform一步到位
    print(le.inverse_transform(label))  # 使用inverse_transform可以逆转
    ## 特征编码
    data_ = data.copy()
    print(data_.head())
    print(OrdinalEncoder().fit(data_.iloc[:, 1:-1]).categories_)  # categories_对应LabelEncoder的接口classes_
    data_.iloc[:, 1:-1] = OrdinalEncoder().fit_transform(data_.iloc[:, 1:-1])
    print(data_.head())
    ## 独特编码，创建哑变量
    print(data.head())
    X = data.iloc[:, 1:-1]
    enc = OneHotEncoder(categories='auto').fit(X)
    result = enc.transform(X).toarray()
    print(result)
    print(OneHotEncoder(categories='auto').fit_transform(X).toarray())  # 依然可以直接一步到位
    print(pd.DataFrame(enc.inverse_transform(result)))  # 依然可以还原
    print(enc.get_feature_names())
    newdata = pd.concat([data, pd.DataFrame(result)], axis=1)
    print(newdata.head())
    newdata.drop(["Sex", "Embarked"], axis=1, inplace=True)
    print(newdata.head())
    newdata.columns = ["Age", "Survived", "Female", "Male", "Embarked_C", "Embarked_Q", "Embarked_S"]
    print(newdata.head())
    ## 类sklearn.preprocessing.LabelBinarizer可以对标签做哑变量，许多算法都可以处理多标签问题（比如说决策树），但是这样的做法在现实中不常见，因此我们在这里就不赘述了。


def binarization_and_segmentation():
    '''
    4、处理连续型特征：二值化与分段
    :return:
    '''
    ## 根据阈值将连续型数据二值化
    data = missing_value()
    data_2 = data.copy()
    X = data_2.iloc[:, 0].values.reshape(-1, 1)  # Binarizer类为特征专用，所以不能使用一维数组
    transformer = Binarizer(threshold=30).fit_transform(X)
    print(transformer)
    data_2.iloc[:, 0] = transformer
    ## 将连续性数据分段(分箱)
    X = data.iloc[:, 0].values.reshape(-1, 1)
    est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')  # 等宽分成三箱，然后顺序编码
    est.fit_transform(X)
    # 查看转换后分的箱：变成了一列中的三箱
    print(set(est.fit_transform(X).ravel()))
    est = KBinsDiscretizer(n_bins=3, encode='onehot', strategy='uniform')  # 等宽变成三箱，然后独热编码
    # 查看转换后分的箱：变成了哑变量
    print(est.fit_transform(X).toarray())


if __name__ == "__main__":
    # 数据无量纲化
    # nondimensionalization()
    # 缺失值处理
    # missing_value()
    # 编码与哑变量
    # coding_and_dummy_variables()
    # 二值化与分段
    binarization_and_segmentation()
