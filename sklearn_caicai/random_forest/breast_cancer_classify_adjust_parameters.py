#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/6/20 0020 上午 10:26
# @Author : West Field
# @File : breast_cancer_classify_adjust_parameters.py

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 10)  # a就是你要设置显示的最大列数参数
pd.set_option('display.max_rows', 10)  # b就是你要设置显示的最大的行数参数
pd.set_option('display.width', 100)  # x就是你要设置的显示的宽度，防止轻易换行


def adjust_parameters():
    # 衡量模型在未知数据上的准确率的指标是泛化误差。泛化误差的背后其实是“偏差-方差困境”。
    # 只有当模型的复杂度刚刚好时才能达到泛化误差最小的目标，模型太简单或太复杂都会导致泛化误差增大。
    # 调参时，根据参数对模型复杂度影响力从大到小调。
    # 有一些参数是用学习曲线来调比较好的，有一些参数是用网格搜索来调比较好的。
    ## 加载数据
    data = load_breast_cancer()
    print(data)
    print(data.data.shape, data.target.shape)
    ## 建模
    rfc = RandomForestClassifier(n_estimators=100, random_state=90)
    score_pre = cross_val_score(rfc, data.data, data.target, cv=10, scoring='accuracy').mean()
    print(score_pre)  # 0.9648809523809524
    ## 调参
    # 画学习曲线调参。第一次的学习曲线，可以先用来帮助我们划定范围，我们取每十个数作为一个阶段，来观察n_estimators的变化如何引起模型整体准确率的变化
    # scorel = []
    # for i in range(0, 200, 10):
    #     rfc = RandomForestClassifier(n_estimators=i + 1,
    #                                  n_jobs=-1,
    #                                  random_state=90)
    #     score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
    #     scorel.append(score)
    # print(max(scorel), (scorel.index(max(scorel)) * 10) + 1)  # 0.9631265664160402 71
    # plt.figure(figsize=[20, 5])
    # plt.plot(range(1, 201, 10), scorel)
    # plt.show()
    # 在确定好的范围内，进一步细化学习曲线
    # scorel = []
    # for i in range(65, 75):
    #     rfc = RandomForestClassifier(n_estimators=i,
    #                                  n_jobs=-1,
    #                                  random_state=90)
    #     score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
    #     scorel.append(score)
    # print(max(scorel), ([*range(65, 75)][scorel.index(max(scorel))]))  # 0.9666353383458647 73
    # plt.figure(figsize=[20, 5])
    # plt.plot(range(65, 75), scorel)
    # plt.show()
    # 网格搜索
    # 调整max_depth
    # param_grid = {'max_depth': np.arange(1, 20, 1)}
    # # 一般根据数据的大小来进行一个试探，小数据集可以采用1~10，或者1~20这样的试探，大型数据来说，我们应该尝试30~50层深度，或许还不足够更应该画出学习曲线，来观察深度对模型的影响
    # rfc = RandomForestClassifier(n_estimators=73, random_state=90)
    # GS = GridSearchCV(rfc, param_grid, cv=10)
    # GS.fit(data.data, data.target)
    # print(GS.best_params_) # {'max_depth': 8}
    # print(GS.best_score_) # 0.9666353383458647
    # 上边准确率没增长，说明限制树的深度来使模型变简单，这种做法没有降低泛化误差，说明之前的模型不在泛化误差图像的右半边。所以调节特征数来使模型变复杂。
    # 调整max_features
    # param_grid = {'max_features': np.arange(5, 30, 1)}
    # rfc = RandomForestClassifier(n_estimators=73, random_state=90)
    # GS = GridSearchCV(rfc, param_grid, cv=10)
    # GS.fit(data.data, data.target)
    # print(GS.best_params_) # {'max_features': 24}
    # print(GS.best_score_) # 0.9666666666666668
    # 上边模型的准确率有所上升，说明使模型变复杂使得模型的泛化误差减小，说明在此之前模型的复杂度处于泛化误差图的左半边部分。
    # 调整min_samples_leaf
    # param_grid = {'min_samples_leaf': np.arange(1, 1 + 10, 1)}
    # rfc = RandomForestClassifier(n_estimators=73, random_state=90, max_features=24)
    # GS = GridSearchCV(rfc, param_grid, cv=10)
    # GS.fit(data.data, data.target)
    # print(GS.best_params_) # {'min_samples_leaf': 1}
    # print(GS.best_score_) # 0.9666666666666668
    # 发现上边使得准确率下降，这种情况下，该参数无需设置，保持默认值就行了
    # 调整min_samples_split
    # param_grid = {'min_samples_split': np.arange(2, 2 + 20, 1)}
    # rfc = RandomForestClassifier(n_estimators=73, random_state=90, max_features=24)
    # GS = GridSearchCV(rfc, param_grid, cv=10)
    # GS.fit(data.data, data.target)
    # print(GS.best_params_)  # {'min_samples_split': 6}
    # print(GS.best_score_) # 0.9701754385964912
    # 发现上边的准确率上升了
    # 调整criterion
    # param_grid = {'criterion':['gini', 'entropy']}
    # rfc = RandomForestClassifier(n_estimators=73, random_state=90, max_features=24, min_samples_split=6)
    # GS = GridSearchCV(rfc, param_grid, cv=10)
    # GS.fit(data.data, data.target)
    # print(GS.best_params_) # {'criterion': 'gini'}
    # print(GS.best_score_) # 0.9701754385964912
    # # 以上模型准确率没变，说明还是保持默认gini即可
    # 调整完毕，总结出模型的最佳参数
    rfc = RandomForestClassifier(n_estimators=73, random_state=90, max_features=24, min_samples_split=6)
    score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
    print(score, score - score_pre)  # 0.9701754385964912 0.005294486215538852
    # 总结：在整个调参过程之中，我们首先调整了n_estimators（无论如何都请先走这一步），然后调整max_depth，通过
    # max_depth产生的结果，来判断模型位于复杂度-泛化误差图像的哪一边，从而选择我们应该调整的参数和调参的方向。


if __name__ == "__main__":
    # 乳腺癌分类任务上调参
    adjust_parameters()
