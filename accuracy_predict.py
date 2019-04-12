# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/12 9:33'

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib


class cal_accuracy:
    def __init__(self, name="result.xlsx"):
        self.file_name = name

    def read_data(self):
        self.data = pd.read_excel("result.xlsx")

    def C_model_cal_C(self):
        data = self.data[(self.data["C收得率"] > 0) | (self.data["C收得率"] < 1)]
        # 获得C收得率
        C_label = data.iloc[:, -2]
        self.label = ["转炉终点温度", "转炉终点C", "钢水净重", "连铸正样C", "钒铁(FeV50-A)", "钒铁(FeV50-B)", "硅铝合金FeAl30Si25",
                      "硅锰面（硅锰渣）", "硅铁(合格块)", "硅铁FeSi75-B", "石油焦增碳剂",
                      "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)", "碳化硅(55%)", "硅钙碳脱氧剂"]
        data_C = data.loc[:, self.label]
        return data_C, C_label

    def C_Mn_model_cal_Mn(self):
        data = self.data[(self.data["Mn收得率"] > 0) | (self.data["Mn收得率"] < 1)]
        Mn_label = data.iloc[:, -1]
        self.label = ["转炉终点温度", "转炉终点C", '钢水净重', "钒铁(FeV50-A)", "钒铁(FeV50-B)", "硅铝合金FeAl30Si25", "硅铝锰合金球",
                      "硅锰面（硅锰渣）", "硅铁(合格块)", "硅铁FeSi75-B", "石油焦增碳剂",
                      "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)", "碳化硅(55%)", "硅钙碳脱氧剂", ]
        # self.label = ["转炉终点温度", "转炉终点C", '钢水净重', '连铸正样C', '连铸正样Mn',
        #               "钒铁(FeV50-A)", "钒铁(FeV50-B)", "硅铝合金FeAl30Si25", "硅铝锰合金球",
        #               "硅锰面（硅锰渣）", "硅铁(合格块)", "硅铁FeSi75-B", "石油焦增碳剂",
        #               "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)", "碳化硅(55%)", "硅钙碳脱氧剂", ]
        # 剔除, '转炉终点Mn'
        data = data.loc[:, self.label]
        return data, Mn_label

    def C_Mn_model_cal_C_Mn(self):
        data = self.data[(self.data["Mn收得率"] > 0) | (self.data["Mn收得率"] < 1)]
        C_Mn_label = data.iloc[:, -3:-1]
        # self.label = ["转炉终点温度", "转炉终点C", '钢水净重', '连铸正样C', '连铸正样Mn',
        #                               "钒铁(FeV50-A)", "钒铁(FeV50-B)", "硅铝合金FeAl30Si25", "硅铝锰合金球",
        #                               "硅锰面（硅锰渣）", "硅铁(合格块)", "硅铁FeSi75-B", "石油焦增碳剂",
        #                               "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)", "碳化硅(55%)", "硅钙碳脱氧剂", ]
        self.label = ["转炉终点温度", "转炉终点C", '钢水净重', "钒铁(FeV50-A)", "钒铁(FeV50-B)", "硅铝合金FeAl30Si25", "硅铝锰合金球",
                      "硅锰面（硅锰渣）", "硅铁(合格块)", "硅铁FeSi75-B", "石油焦增碳剂",
                      "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)", "碳化硅(55%)", "硅钙碳脱氧剂", ]
        # 剔除, '转炉终点Mn'
        data = data.loc[:, self.label]
        return data, C_Mn_label

    def Mn_model_cal_Mn(self):
        # 获得Mn收得率
        data = self.data[(self.data["Mn收得率"] > 0) | (self.data["Mn收得率"] < 1)]
        Mn_label = data.iloc[:, -1]
        self.label = ["转炉终点温度", "钢水净重", "钒铁(FeV50-B)", "硅锰面（硅锰渣）",
                      "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)"]
        # 剔除, "连铸正样Mn"
        data_Mn = data.loc[:, self.label]
        return data_Mn, Mn_label

    def train(self, C=False, Mn=False, C_Mn_cal_Mn_C=True, C_Mn_cal_Mn=False):
        data = None
        label = None
        if C:
            data, label = self.C_model_cal_C()
        if Mn:
            data, label = self.Mn_model_cal_Mn()  # 0.0301574843803
        if C_Mn_cal_Mn_C:
            data, label = self.C_Mn_model_cal_C_Mn()
        if C_Mn_cal_Mn:
            data, label = self.C_Mn_model_cal_Mn()  # 0.020663763357
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
        self.std = StandardScaler()
        self.std.fit(X_train)
        X_train = self.std.transform(X_train)
        X_test = self.std.transform(X_test)

        # model = RandomForestRegressor()  # 0.00178279872547 0.00130315829688
        self.model = LinearRegression()  # 8.36667068081e-05  0.00016716587287
        # model = Ridge() # 7.8575505755e-05              0.000232658506956
        # model = Lasso() # 0.0046516170403                  0.00238670854501
        # model = SVR()  # 0.00328161901572                 0.00307726323407
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # 模型的存储
        # joblib.dump(self.model, "Mn.m")
        print(np.sqrt(mean_squared_error(y_pred, y_test)))

    def run(self):
        self.read_data()
        self.train()

    def fill_Mn(self):
        data = self.data.iloc[192:, :]
        data_input = data.loc[:, self.label]
        data_input = self.std.transform(data_input)
        res = self.model.predict(data_input).ravel()
        for i in np.arange(len(res)):
            self.data.loc[192 + i:, "Mn收得率"] = res[i]
        self.data.to_excel("fill_Mn_result.xlsx", index=False)
    def fill_C_Mn(self):
        data = self.data.iloc[193, :]
        data_input = data.loc[:, self.label]
        data_input = self.std.transform(data_input)
        res = self.model.predict(data_input)
        print(res)
        # for i in np.arange(len(res)):
        #     self.data.loc[192 + i:, "Mn收得率"] = res[i]
        #     self.data.loc[192 + i:, "Mn收得率"] = res[i]
        # self.data.to_excel("fill_Mn_C_result.xlsx", index=False)

if __name__ == '__main__':
    item = cal_accuracy()
    item.run()
    # item.fill_Mn()
    item.fill_C_Mn()