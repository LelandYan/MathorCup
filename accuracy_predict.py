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


class cal_accuracy:
    def __init__(self, name="result.xlsx"):
        self.file_name = name

    def read_data(self):
        self.data = pd.read_excel("result.xlsx")

    def C_model(self):
        self.data = self.data[(self.data["C收得率"] > 0) | (self.data["C收得率"] < 1)]
        # 获得C收得率
        self.C_label = self.data.iloc[:, -2]
        self.data_C = self.data.loc[:,
                      ["转炉终点温度", "转炉终点C", "钢水净重", "连铸正样C", "钒铁(FeV50-A)", "钒铁(FeV50-B)", "硅铝合金FeAl30Si25",
                       "硅锰面（硅锰渣）", "硅铁(合格块)", "硅铁FeSi75-B", "石油焦增碳剂",
                       "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)", "碳化硅(55%)", "硅钙碳脱氧剂"]]
        return self.data_C, self.C_label

    def C_Mn_model(self):
        self.data = self.data[(self.data["Mn收得率"] > 0) | (self.data["Mn收得率"] < 1)]
        self.label = self.data.iloc[:, -3:-1]
        self.data = self.data.loc[:, ["转炉终点温度", "转炉终点C", '转炉终点Mn', '钢水净重', '连铸正样C', '连铸正样Mn',
                                         "钒铁(FeV50-A)", "钒铁(FeV50-B)", "硅铝合金FeAl30Si25", "硅铝锰合金球",
                                         "硅锰面（硅锰渣）", "硅铁(合格块)", "硅铁FeSi75-B", "石油焦增碳剂",
                                         "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)", "碳化硅(55%)", "硅钙碳脱氧剂", ]]
        return self.data, self.label

    def Mn_model(self):
        # 获得Mn收得率
        self.data = self.data[(self.data["Mn收得率"] > 0) | (self.data["Mn收得率"] < 1)]
        self.Mn_label = self.data.iloc[:, -1]
        self.data_Mn = self.data.loc[:, ["转炉终点温度", "转炉终点Mn", "钢水净重", "连铸正样Mn", "钒铁(FeV50-B)", "硅锰面（硅锰渣）",
                                         "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)"]]
        return self.data_Mn, self.Mn_label

    def train(self, C=False, C_Mn=True):
        data = None
        label = None
        if C:
            data, label = self.C_model()
        if not C:
            data, label = self.Mn_model()
        if C_Mn:
            data, label = self.C_Mn_model()
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
        std = StandardScaler()
        std.fit(X_train)
        X_train = std.transform(X_train)
        X_test = std.transform(X_test)

        #model = RandomForestRegressor()  # 0.00178279872547 0.00130315829688
        model = LinearRegression()  # 8.36667068081e-05  0.00016716587287
        # model = Ridge() # 7.8575505755e-05              0.000232658506956
        # model = Lasso() # 0.0046516170403                  0.00238670854501
        # model = SVR()  # 0.00328161901572                 0.00307726323407
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(mean_squared_error(y_pred, y_test))

    def run(self):
        self.read_data()
        self.train()


if __name__ == '__main__':
    item = cal_accuracy()
    item.run()
