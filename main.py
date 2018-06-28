# -*-coding:utf-8 -*-

from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
from numpy import arange
from matplotlib import pyplot
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error

boston = load_boston()
boston_X = pd.DataFrame(data=boston['data'], columns=boston['feature_names'])
boston_Y = pd.DataFrame(data=boston['target'], columns=['target'])
dataset = pd.concat((boston_X, boston_Y), axis=1)

# 查看最开始的30条记录
set_option('display.line_width', 120)
# print(dataset.loc[:30, :], '\n')

# 描述性统计信息
set_option('precision', 1)
# print(dataset.describe(), '\n')

# 查看两两之间关联关系,pearson系数
set_option('precision', 2)
# print(dataset.corr(method='pearson'), '\n')
# print((dataset.corr(method='pearson')>0.7) | (dataset.corr(method='pearson')<-0.7), '\n')

# 数据可视化
# 直方图
# dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
# pyplot.show()

# 密度图
# dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, fontsize=1)
# pyplot.show()

# 箱线图
# dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, fontsize=8)
# pyplot.show()

# 散点矩阵图
# scatter_matrix(dataset)
# pyplot.show()

# 相关矩阵图
# fig = pyplot.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
# fig.colorbar(cax)
# ticks = np.arange(0, 14, 1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(dataset.columns)
# ax.set_yticklabels(dataset.columns)
# pyplot.show()

# 分离数据集
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(boston_X, dataset['target'],
                                                                test_size=validation_size, random_state=seed)
# 评估算法 —— 评估标准
num_folds = 10  # 10折交叉验证
seed = 7
scoring = 'neg_mean_squared_error'

# 线性算法：线性回归（LR）、套索回归（LASSO）和弹性网络回归（EN）。
# 非线性算法：分类与回归树（CART）、支持向量机（SVM）和K近邻算法（KNN）。

# 评估算法 - baseline
models = {}
models['LR'] = LinearRegression()
models['LASSO'] = Lasso()
models['EN'] = ElasticNet()
models['KNN'] = KNeighborsRegressor()
models['CART'] = DecisionTreeRegressor()
models['SVM'] = SVR()

# 对所有的算法使用默认参数，并比较算法的准确度，此处比较的是均方误差的均值和标准方差。代码如下
# 评估算法
# print(boston_Y)
# results = []
# for key in models:
#     kfold = KFold(n_splits=num_folds, random_state=seed)
#     cv_result = cross_val_score(models[key], X_train, Y_train, cv=kfold, scoring=scoring)
#     print(cv_result)
#     results.append(cv_result)
#     print('%s: %f (%f)' % (key, cv_result.mean(), cv_result.std()))
#

# 评估算法——正态化数据,均值为0，方差为1
# 对数据正态化时，为了防止数据泄露，采用Pipeline来正态化数据和对模型进行评估
# pipelines = {}
# pipelines['ScalerLR'] = Pipeline([('Scale', StandardScaler()), ('LR', LinearRegression())])
# pipelines['ScalerLASSO'] = Pipeline([('Scaler', StandardScaler()), ('LASSO', Lasso())])
# pipelines['ScalerEN'] = Pipeline([('Scaler', StandardScaler()), ('EN', ElasticNet())])
# pipelines['ScalerKNN'] = Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor())])
# pipelines['ScalerCART'] = Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeRegressor())])
# pipelines['ScalerSVM'] = Pipeline([('Scaler', StandardScaler()), ('SVM', SVR())])
# results = []
# for key in pipelines:
#     kfold = KFold(n_splits=num_folds, random_state=seed)
#     cv_result = cross_val_score(pipelines[key], X_train, Y_train, cv=kfold, scoring=scoring)
#     results.append(cv_result)
#     print('%s: %f (%f)' % (key, cv_result.mean(), cv_result.std()))

# 调参改善算法——KNN, 网格搜索法
# scaler = StandardScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)
# param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}
# model = KNeighborsRegressor()
# kfold = KFold(n_splits=num_folds, random_state=seed)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
# grid_result = grid.fit(X=rescaledX, y=Y_train)
# print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))
# cv_results = zip(grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['std_test_score'],
#                  grid_result.cv_results_['params'])
# for mean, std, param in cv_results:
#     print('%f (%f) with %r' % (mean, std, param))

# 集成算法
# ensembles = {}
# ensembles['ScaledAB'] = Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostRegressor())])
# ensembles['ScaledAB-KNN'] = Pipeline([('Scaler', StandardScaler()),
#                                       ('ABKNN', AdaBoostRegressor(base_estimator=KNeighborsRegressor(n_neighbors=3)))])
# ensembles['ScaledAB-LR'] = Pipeline([('Scaler', StandardScaler()), ('ABLR', AdaBoostRegressor(LinearRegression()))])
# ensembles['ScaledRFR'] = Pipeline([('Scaler', StandardScaler()), ('RFR', RandomForestRegressor())])
# ensembles['ScaledETR'] = Pipeline([('Scaler', StandardScaler()), ('ETR', ExtraTreesRegressor())])
# ensembles['ScaledGBR'] = Pipeline([('Scaler', StandardScaler()), ('RBR', GradientBoostingRegressor())])
#
# results = []
# for key in ensembles:
#     kfold = KFold(n_splits=num_folds, random_state=seed)
#     cv_result = cross_val_score(ensembles[key], X_train, Y_train, cv=kfold, scoring=scoring)
#     results.append(cv_result)
#     print('%s: %f (%f)' % (key, cv_result.mean(), cv_result.std()))
#
# # 集成算法——箱线图
# fig = pyplot.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# pyplot.boxplot(results)
# ax.set_xticklabels(ensembles.keys())
# pyplot.show()

# 集成算法GBM——调参
# scaler = StandardScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)
# param_grid = {'n_estimators': [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900]}
# model = GradientBoostingRegressor()
# kfold = KFold(n_splits=num_folds, random_state=seed)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
# grid_result = grid.fit(X=rescaledX, y=Y_train)
# print('最优：%s 使用%s' % (grid_result.best_score_,
# grid_result.best_params_))
#
# # 集成算法ET——调参
# scaler = StandardScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)
# param_grid = {'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
# model = ExtraTreesRegressor()
# kfold = KFold(n_splits=num_folds, random_state=seed)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
# grid_result = grid.fit(X=rescaledX, y=Y_train)
# print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))

# 确定最终模型为GBR
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
gbr = GradientBoostingRegressor(n_estimators=700)
gbr.fit(X=rescaledX, y=Y_train)
# 评估算法模型
rescaledX_validation = scaler.transform(X_validation)
predictions = gbr.predict(rescaledX_validation)
print(mean_squared_error(Y_validation, predictions))

