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
from pandas.plotting import scatter_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
class cal_corr:
    def __init__(self, name="result.xlsx"):
        self.name = name
        self.pearsonr_rate = []
        self.mutual_info_rate = []

    def read_data(self):
        # 读取数据
        self.data = pd.read_excel(self.name)
        self.data = self.data.iloc[:192, :]

    def plot_corr(self, C=True):
        mpl.rcParams['font.sans-serif'] = [u'SimHei']
        mpl.rcParams['axes.unicode_minus'] = False
        self.read_data()
        if C:
            col = ["转炉终点温度", "转炉终点C", "钢水净重", "连铸正样C", "钒铁(FeV50-A)", "钒铁(FeV50-B)", "硅铝合金FeAl30Si25",
                   "硅锰面（硅锰渣）", "硅铁(合格块)", "硅铁FeSi75-B", "石油焦增碳剂",
                   "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)", "碳化硅(55%)", "硅钙碳脱氧剂"]
            scatter_matrix(self.data.loc[:, col], figsize=(25, 25))
        plt.show()

    def pearsonr_cal(self,C_label=False):
        if C_label:
            self.label = ["转炉终点温度", "转炉终点C", "钢水净重", "连铸正样C", "钒铁(FeV50-A)", "钒铁(FeV50-B)", "硅铝合金FeAl30Si25",
                          "硅锰面（硅锰渣）", "硅铁(合格块)", "硅铁FeSi75-B", "石油焦增碳剂",
                          "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)", "碳化硅(55%)", "硅钙碳脱氧剂"]
            self.data = self.data[(self.data["C收得率"] > 0) | (self.data["C收得率"] < 1)]
            self.data = self.data.loc[:, self.label]
            self.data = StandardScaler().fit_transform(self.data)
            label = self.data[:, -2]
        else:
            self.label = ["转炉终点温度", "转炉终点Mn", "钢水净重", "连铸正样Mn", "钒铁(FeV50-B)", "硅锰面（硅锰渣）",
                          "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)"]
            self.data = self.data[(self.data["Mn收得率"] > 0) | (self.data["Mn收得率"] < 1)]
            self.data = self.data.loc[:, self.label]
            self.data = StandardScaler().fit_transform(self.data)
            label = self.data[:, -1]
        for i in np.arange(self.data.shape[1]):
            self.pearsonr_rate.append(pearsonr(label, self.data[:, i])[1])
        self.pearsonr_rate = zip(self.label, self.pearsonr_rate)
        #self.pearsonr_rate = np.array(self.pearsonr_rate)

    def mutual_info(self, C_label=False):
        if C_label:
            self.label = ["转炉终点温度", "转炉终点C", "钢水净重", "连铸正样C", "钒铁(FeV50-A)", "钒铁(FeV50-B)", "硅铝合金FeAl30Si25",
                          "硅锰面（硅锰渣）", "硅铁(合格块)", "硅铁FeSi75-B", "石油焦增碳剂",
                          "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)", "碳化硅(55%)", "硅钙碳脱氧剂"]
            self.data = self.data[(self.data["C收得率"] > 0) | (self.data["C收得率"] < 1)]
            self.data = self.data.loc[:, self.label]
            self.data = StandardScaler().fit_transform(self.data)
            label = self.data[:, -2]
        else:
            self.label = ["转炉终点温度", "转炉终点Mn", "钢水净重", "连铸正样Mn", "钒铁(FeV50-B)", "硅锰面（硅锰渣）",
                          "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)"]
            self.data = self.data[(self.data["Mn收得率"] > 0) | (self.data["Mn收得率"] < 1)]
            self.data = self.data.loc[:, self.label]
            self.data = StandardScaler().fit_transform(self.data)
            label = self.data[:, -1]
        for i in np.arange(self.data.shape[1]):
            self.mutual_info_rate.append(mr.mutual_info_score(label, self.data[:, i]))
        self.mutual_info_rate = zip(self.label, self.mutual_info_rate)
        # self.mutual_info_rate = np.array(self.mutual_info_rate)

    def run(self, pearsonr_cal=True,mutual_info=True):
        self.read_data()
        if pearsonr_cal:
            self.pearsonr_cal()
            return self.pearsonr_rate
        if mutual_info:
            self.mutual_info()
            return list(self.mutual_info_rate)


if __name__ == '__main__':
    item = cal_corr()
    # item.plot_corr()
    for name,value in item.run():
        print(name,":",value)
