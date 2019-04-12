# _*_ coding: utf-8 _*_
__author__ = 'Yxp'
__date__ = '2019/4/11 17:40'
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RandomizedLasso
from sklearn.feature_selection import RFE, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics as mr


class cal_corr:
    def __init__(self, name="result.xlsx"):
        self.name = name
        self.pearsonr_rate = []
        self.mutual_info_rate = []

    def read_data(self):
        # 读取数据
        self.data = pd.read_excel("result.xlsx")
        self.label = self.data.columns.values
        # 获得C收得率
        self.C_label = self.data.iloc[:, -3]
        # 获得Mn收得率
        self.Mn_label = self.data.iloc[:, -1]
        # 获取属性
        self.data = self.data.iloc[:, :-4]
        self.data = StandardScaler().fit_transform(self.data)
    def pearsonr_cal(self):
        for i in np.arange(self.data.shape[1]):
            self.pearsonr_rate.append(pearsonr(self.C_label, self.data.iloc[:, i])[1])
        self.pearsonr_rate = np.array(self.pearsonr_rate)

    def corr_cal(self):
        self.f, pval = f_regression(self.C_label, self.data, center=True)

    def mutual_info(self,C_label=True):
        if C_label:
            label = self.C_label
        else:
            label = self.Mn_label
        for i in np.arange(self.data.shape[1]):
            self.mutual_info_rate.append(mr.mutual_info_score(label, self.data[:, i]))
        self.mutual_info_rate = zip(self.label,self.mutual_info_rate)
        #self.mutual_info_rate = np.array(self.mutual_info_rate)

    def run(self, pearsonr_cal=False, corr_cal=False, mutual_info=True):
        self.read_data()
        if pearsonr_cal:
            self.pearsonr_cal()
            return self.pearsonr_rate
        if corr_cal:
            self.corr_cal()
            return self.f
        if mutual_info:
            self.mutual_info()
            return list(self.mutual_info_rate)


if __name__ == '__main__':
    pass
    item = cal_corr()
    print(item.run())
