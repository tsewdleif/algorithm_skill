#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/6/28 0028 上午 11:51
# @Author : West Field
# @File : ranking_card_classify.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestRegressor as RFR
from imblearn.over_sampling import SMOTE  # imblearn专门用来处理不平衡数据集的库，在处理样本不均衡问题中性能高过sklearn很多
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy
import scikitplot as skplt

pd.set_option('display.max_columns', 20)  # a就是你要设置显示的最大列数参数
pd.set_option('display.max_rows', 20)  # b就是你要设置显示的最大的行数参数
pd.set_option('display.width', 1000)  # x就是你要设置的显示的宽度，防止轻易换行


def ranking_card_classifier():
    '''
    在银行借贷场景中，评分卡是一种以分数形式来衡量一个客户的信用风险大小的手段
    :return:
    '''
    ## 导入评分卡分类数据集，二分类数据集，第一列是索引，第二列是标签。
    data = pd.read_csv(r".\rankingcard.csv", index_col=0)
    ## 探索数据与数据预处理
    # 观察数据类型
    print(data.head())
    # 观察数据结构
    print(data.shape)  # (150000, 11)
    print(data.info())
    # 1.去除重复样本
    data.drop_duplicates(inplace=True)
    print(data.info())
    data.index = range(data.shape[0])  # 重复样本删除之后千万不要忘记，恢复索引
    print(data.info())
    # 2.填补缺失值
    print(data.isnull().sum() / data.shape[0])  # 探索缺失值
    data["NumberOfDependents"].fillna(int(data["NumberOfDependents"].mean()), inplace=True)  # 对缺失值2.5%的特征使用均值填充
    # 对于缺失值19.5%的特征，使用随机森林填补缺失值
    X = data.iloc[:, 1:]
    y = data["SeriousDlqin2yrs"]
    print(X.shape)  # (149391, 10)
    y_pred = fill_missing_rf(X, y, "MonthlyIncome")
    data.loc[data.loc[:, "MonthlyIncome"].isnull(), "MonthlyIncome"] = y_pred
    print(data.info())
    # 3.描述性统计处理异常值：一般箱线图或者法则来找到异常值，但是在银行数据中这里出现的异常值这些是我们要重点研究观察的，我们希望排除的“异常值”不是一些超高或超低的数字，而是一些不符合常理的数据。
    # print(data.describe([0.01, 0.1, 0.25, .5, .75, .9, .99]).T)  # 描述性统计
    print((data["age"] == 0).sum())  # 年龄等于0的样本个数
    data = data[data["age"] != 0]  # 删除年龄为0的1个样本
    print(data[data.loc[:, "NumberOfTimes90DaysLate"] > 90].count())  # 两年内出现90天违约的次数
    data = data[data.loc[:, "NumberOfTimes90DaysLate"] < 90]  # 删除大于90次的违约次数异常值
    # 恢复索引
    data.index = range(data.shape[0])
    print(data.info())
    # print(data.describe([0.01, 0.1, 0.25, .5, .75, .9, .99]).T)  # 描述性统计
    # 4.样本不均衡问题
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    print(y.value_counts())
    n_sample = X.shape[0]
    n_1_sample = y.value_counts()[1]
    n_0_sample = y.value_counts()[0]
    print("%s; %s; %s" % (n_sample, n_1_sample / n_sample, n_0_sample / n_sample))
    # 逻辑回归中使用最多的是上采样方法来平衡样本
    sm = SMOTE(random_state=42)  # 实例化
    X, y = sm.fit_sample(X, y)
    n_sample_ = X.shape[0]
    pd.Series(y).value_counts()
    n_1_sample = pd.Series(y).value_counts()[1]
    n_0_sample = pd.Series(y).value_counts()[0]
    print("%s; %s; %s" % (n_sample_, n_1_sample / n_sample_, n_0_sample / n_sample_))  # 如此，我们就实现了样本平衡，样本量也增加了
    # 5.分训练集和测试集
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    X_train, X_vali, Y_train, Y_vali = train_test_split(X, y, test_size=0.3,
                                                        random_state=420)  # 训练数据是用来建模的，验证数据是用来检测模型的效果的
    model_data = pd.concat([Y_train, X_train], axis=1)
    model_data.index = range(model_data.shape[0])
    model_data.columns = data.columns
    vali_data = pd.concat([Y_vali, X_vali], axis=1)
    vali_data.index = range(vali_data.shape[0])
    vali_data.columns = data.columns
    model_data.to_csv(r".\model_data.csv")
    vali_data.to_csv(r".\vali_data.csv")


def fenxiang():
    '''

    :return:
    '''
    ## 导入评分卡分类数据集
    model_data = pd.read_csv(r".\model_data.csv", index_col=0)
    ## 等频分箱
    # pd.qcut，基于分位数的分箱函数，本质是将连续型变量离散化 只能够处理一维数据。返回箱子的上限和下限。
    # 参数q：要分箱的个数。参数retbins=True来要求同时返回结构为索引为样本索引，元素为分到的箱子的Series。
    # 在这里时让model_data新添加一列叫做“分箱”，这一列其实就是每个样本所对应的箱子。updown是所有箱子的上限和下限。
    model_data["qcut"], updown = pd.qcut(model_data["age"], retbins=True, q=20)
    print(model_data["qcut"])
    print(updown)
    # 统计每个分箱中0和1的数量
    coount_y0 = model_data[model_data["SeriousDlqin2yrs"] == 0].groupby(by="qcut").count()["SeriousDlqin2yrs"]
    coount_y1 = model_data[model_data["SeriousDlqin2yrs"] == 1].groupby(by="qcut").count()["SeriousDlqin2yrs"]
    num_bins = [*zip(updown, updown[1:], coount_y0, coount_y1)]  # 每个区间的上界，下界，0出现的次数，1出现的次数
    print(num_bins)

    # 定义WOE和IV函数
    def get_woe(num_bins):
        # 通过 num_bins 数据计算  woe
        # 证据权重WOE是对一个箱子来说的，WOE越大，代表了这个箱子里的优质客户越多。
        columns = ["min", "max", "count_0", "count_1"]
        df = pd.DataFrame(num_bins, columns=columns)
        df["total"] = df.count_0 + df.count_1
        df["percentage"] = df.total / df.total.sum()
        df["bad_rate"] = df.count_1 / df.total
        df["good%"] = df.count_0 / df.count_0.sum()
        df["bad%"] = df.count_1 / df.count_1.sum()
        df["woe"] = np.log(df["good%"] / df["bad%"])
        return df

    def get_iv(df):
        # 计算IV值
        # IV是对整个特征来说的，IV代表特征上的信息量以及这个特征对模型的贡献
        rate = df["good%"] - df["bad%"]
        iv = np.sum(rate * df.woe)
        return iv

    ## 卡方检验(用来计算相似性)，合并箱体，画出IV曲线(即合并不同个数个分箱后的年龄特征对应的IV值变化)
    num_bins_ = num_bins.copy()
    IV = []
    axisx = []
    while len(num_bins_) > 2:
        pvs = []
        # 获取num_bins_两两分箱之间的卡方检验的置信度（或卡方值），即分箱内的0和1的标签的个数的卡方。
        for i in range(len(num_bins_) - 1):
            x1 = num_bins_[i][2:]
            x2 = num_bins_[i + 1][2:]
            # 计算两个数组x1和x2之间的卡方值。
            pv = scipy.stats.chi2_contingency([x1, x2])[1]  # [1]返回P-value，如果是[0]则返回卡方值
            pvs.append(pv)
        # 通过 p 值进行处理。合并 p 值最大的相邻两组分箱。
        i = pvs.index(max(pvs))
        num_bins_[i:i + 2] = [(num_bins_[i][0], num_bins_[i + 1][1], num_bins_[i][2] + num_bins_[i + 1][2],
                               num_bins_[i][3] + num_bins_[i + 1][3])]
        bins_df = get_woe(num_bins_)
        axisx.append(len(num_bins_))
        IV.append(get_iv(bins_df))
    plt.figure()
    plt.plot(axisx, IV)
    plt.xticks(axisx)
    plt.xlabel("number of box")
    plt.ylabel("IV")
    plt.show()  # 由图可以看到在分箱为6时发生转折，所以选择分箱为6

    ## 用最佳分箱个数分箱，并验证分箱结果
    def get_bin(num_bins_, n):
        # 将合并箱体的部分定义为函数，并实现分箱
        while len(num_bins_) > n:
            pvs = []
            for i in range(len(num_bins_) - 1):
                x1 = num_bins_[i][2:]
                x2 = num_bins_[i + 1][2:]
                pv = scipy.stats.chi2_contingency([x1, x2])[1]
                pvs.append(pv)
            i = pvs.index(max(pvs))
            num_bins_[i:i + 2] = [(num_bins_[i][0], num_bins_[i + 1][1], num_bins_[i][2] + num_bins_[i + 1][2],
                                   num_bins_[i][3] + num_bins_[i + 1][3])]
            return num_bins_

    afterbins = get_bin(num_bins, 6)  # 设置分箱合并后的箱子为6个
    print(afterbins)
    # 对所有特征进行分箱选择
    for i in model_data.columns[1:-1]:
        print(i)
        graphforbestbin(model_data, i, "SeriousDlqin2yrs", n=2, q=20)
    ## 可以发现，不是所有的特征都可以使用这个分箱函数，比如说有的特征，像家人数量，就无法分出20组。于是将可以分箱的特征放出来单独分组，不能自动分箱的变量自己观察然后手写：
    auto_col_bins = {"RevolvingUtilizationOfUnsecuredLines": 6, "age": 5,
                     "DebtRatio": 4, "MonthlyIncome": 3,
                     "NumberOfOpenCreditLinesAndLoans": 5}
    # 不能使用自动分箱的变量
    hand_bins = {"NumberOfTime30-59DaysPastDueNotWorse": [0, 1, 2, 13]
        , "NumberOfTimes90DaysLate": [0, 1, 2, 17]
        , "NumberRealEstateLoansOrLines": [0, 1, 2, 4, 54]
        , "NumberOfTime60-89DaysPastDueNotWorse": [0, 1, 2, 8]
        , "NumberOfDependents": [0, 1, 2, 3]}
    # 保证区间覆盖使用 np.inf替换最大值，用-np.inf替换最小值
    hand_bins = {k: [-np.inf, *v[:-1], np.inf] for k, v in hand_bins.items()}
    # 接下来对所有特征按照选择的箱体个数和手写的分箱范围进行分箱：
    bins_of_col = {}
    # 生成自动分箱的分箱区间和分箱后的 IV 值
    for col in auto_col_bins:
        bins_df = graphforbestbin(model_data, col
                                  , "SeriousDlqin2yrs"
                                  , n=auto_col_bins[col]
                                  # 使用字典的性质来取出每个特征所对应的箱的数量
                                  , q=20
                                  , graph=False)
        bins_list = sorted(set(bins_df["min"]).union(bins_df["max"]))
        # 保证区间覆盖使用 np.inf 替换最大值 -np.inf 替换最小值
        bins_list[0], bins_list[-1] = -np.inf, np.inf
        bins_of_col[col] = bins_list
    # 合并手动分箱数据
    bins_of_col.update(hand_bins)
    print(bins_of_col)

    ## 我们现在已经有了我们的箱子，接下来我们要做的是计算各箱的WOE，并且把WOE替换到我们的原始数据
    # model_data中，因为我们将使用WOE覆盖后的数据来建模，我们希望获取的是”各个箱”的分类结果，即评分卡上各个评分项目的分类结果。
    def get_woe(df, col, y, bins):
        df = df[[col, y]].copy()
        df["cut"] = pd.cut(df[col], bins)
        bins_df = df.groupby("cut")[y].value_counts().unstack()
        woe = bins_df["woe"] = np.log((bins_df[0] / bins_df[0].sum()) / (bins_df[1] / bins_df[1].sum()))
        return woe

    # 将所有特征的WOE存储到字典当中
    woeall = {}
    for col in bins_of_col:
        woeall[col] = get_woe(model_data, col, "SeriousDlqin2yrs", bins_of_col[col])
    print(woeall)
    ## 接下来，把所有WOE映射到原始数据中：
    # 不希望覆盖掉原本的数据，创建一个新的DataFrame，索引和原始数据model_data一模一样
    model_woe = pd.DataFrame(index=model_data.index)
    # 对所有特征操作可以写成：
    for col in bins_of_col:
        model_woe[col] = pd.cut(model_data[col], bins_of_col[col]).map(woeall[col])
    # 将标签补充到数据中
    model_woe["SeriousDlqin2yrs"] = model_data["SeriousDlqin2yrs"]
    # 这就是我们的建模数据了
    print(model_woe.head())
    ## 终于弄完了训练集，接下来要处理测试集，在已经有分箱的情况下，测试集的处理就非常简单了，只需要将已经计算好的WOE映射到测试集中去就可以了：
    vali_data = pd.read_csv(r".\vali_data.csv", index_col=0)
    vali_woe = pd.DataFrame(index=vali_data.index)
    for col in bins_of_col:
        vali_woe[col] = pd.cut(vali_data[col], bins_of_col[col]).map(woeall[col])
    vali_woe["SeriousDlqin2yrs"] = vali_data["SeriousDlqin2yrs"]
    vali_X = vali_woe.iloc[:, :-1]
    vali_y = vali_woe.iloc[:, -1]
    ## 开始建模
    X = model_woe.iloc[:, :-1]
    y = model_woe.iloc[:, -1]
    lr = LR().fit(X, y)
    print(lr.score(vali_X, vali_y))  # 0.7880731310424045
    # 返回的结果一般，我们可以试着使用C和max_iter的学习曲线把逻辑回归的效果调上去
    score = []
    for i in np.linspace(0.01, 1, 20):
        lr = LR(solver='liblinear', C=i).fit(X, y)
        score.append(lr.score(vali_X, vali_y))
    plt.figure()
    plt.plot(np.linspace(0.01, 1, 20), score)
    plt.show()
    print(lr.n_iter_)  # [5]
    score = []
    for i in [1, 2, 3, 4, 5, 6]:
        lr = LR(solver='liblinear', C=0.025, max_iter=i).fit(X, y)
        score.append(lr.score(vali_X, vali_y))
    plt.figure()
    plt.plot([1, 2, 3, 4, 5, 6], score)
    plt.show()
    # 尽管从准确率来看，我们的模型效果属于一般，但我们可以来看看ROC曲线上的结果
    vali_proba_df = pd.DataFrame(lr.predict_proba(vali_X))
    skplt.metrics.plot_roc(vali_y, vali_proba_df,
                           plot_micro=False, figsize=(6, 6), plot_macro=False)
    plt.show()
    # 生成各个特征对应的分箱的分值，即评分卡，来根据该评分卡对用户进行打分
    for i, col in enumerate(X.columns):
        # 评分卡中的分数计算公式
        score = woeall[col] * (-20 * lr.coef_[0][i])  # woeall[col]是某个特定的违约概率下的预期分值。20表示指定的违约概率翻倍的分数。
        score.name = "Score"
        score.index.name = col
        score.to_csv("./ScoreData.csv", header=True, mode="a")


def graphforbestbin(DF, X, Y, n=5, q=20, graph=True):
    '''
    将选取最佳分箱个数的过程包装为函数
    自动最优分箱函数，基于卡方检验的分箱
    参数：
    DF: 需要输入的数据
    X: 需要分箱的列名
    Y: 分箱数据对应的标签 Y 列名
    n: 保留分箱个数
    q: 初始分箱的个数
    graph: 是否要画出IV图像
    区间为前开后闭 (]
    :return:
    '''
    DF = DF[[X, Y]].copy()
    DF["qcut"], bins = pd.qcut(DF[X], retbins=True, q=q, duplicates="drop")
    coount_y0 = DF.loc[DF[Y] == 0].groupby(by="qcut").count()[Y]
    coount_y1 = DF.loc[DF[Y] == 1].groupby(by="qcut").count()[Y]
    num_bins = [*zip(bins, bins[1:], coount_y0, coount_y1)]
    for i in range(q):
        if 0 in num_bins[0][2:]:
            num_bins[0:2] = [
                (num_bins[0][0], num_bins[1][1], num_bins[0][2] + num_bins[1][2], num_bins[0][3] + num_bins[1][3])]
            continue
        for i in range(len(num_bins)):
            if 0 in num_bins[i][2:]:
                num_bins[i - 1:i + 1] = [(num_bins[i - 1][0], num_bins[i][1], num_bins[i - 1][2] + num_bins[i][2],
                                          num_bins[i - 1][3] + num_bins[i][3])]
                break
        else:
            break

    def get_woe(num_bins):
        columns = ["min", "max", "count_0", "count_1"]
        df = pd.DataFrame(num_bins, columns=columns)
        df["total"] = df.count_0 + df.count_1
        df["percentage"] = df.total / df.total.sum()
        df["bad_rate"] = df.count_1 / df.total
        df["good%"] = df.count_0 / df.count_0.sum()
        df["bad%"] = df.count_1 / df.count_1.sum()
        df["woe"] = np.log(df["good%"] / df["bad%"])
        return df

    def get_iv(df):
        rate = df["good%"] - df["bad%"]
        iv = np.sum(rate * df.woe)
        return iv

    IV = []
    axisx = []
    bins_df = None
    while len(num_bins) > n:
        pvs = []
        for i in range(len(num_bins) - 1):
            x1 = num_bins[i][2:]
            x2 = num_bins[i + 1][2:]
            pv = scipy.stats.chi2_contingency([x1, x2])[1]
            pvs.append(pv)
        i = pvs.index(max(pvs))
        num_bins[i:i + 2] = [(num_bins[i][0], num_bins[i + 1][1], num_bins[i][2] + num_bins[i + 1][2],
                              num_bins[i][3] + num_bins[i + 1][3])]
        bins_df = pd.DataFrame(get_woe(num_bins))
        axisx.append(len(num_bins))
        IV.append(get_iv(bins_df))
    if graph:
        plt.figure()
        plt.plot(axisx, IV)
        plt.xticks(axisx)
        plt.xlabel("number of box")
        plt.ylabel("IV")
        plt.show()
    return bins_df


def fill_missing_rf(X, y, to_fill):
    """
    使用随机森林填补一个特征的缺失值的函数
    参数：
    X：要填补的特征矩阵
    y：完整的，没有缺失值的标签
    to_fill：字符串，要填补的那一列的名称
    """
    # 构建我们的新特征矩阵和新标签
    df = X.copy()
    fill = df.loc[:, to_fill]
    df = pd.concat([df.loc[:, df.columns != to_fill], pd.DataFrame(y)], axis=1)
    # 找出我们的训练集和测试集
    Ytrain = fill[fill.notnull()]
    Ytest = fill[fill.isnull()]
    Xtrain = df.iloc[Ytrain.index, :]
    Xtest = df.iloc[Ytest.index, :]
    # 用随机森林回归来填补缺失值
    rfr = RFR(n_estimators=100)
    rfr = rfr.fit(Xtrain, Ytrain)
    Ypredict = rfr.predict(Xtest)
    return Ypredict


if __name__ == "__main__":
    # 评分卡分类
    # ranking_card_classifier()
    # 分箱
    fenxiang()
