# _*_ coding: utf-8 _*_
__author__ = 'Yxp'
__date__ = '2019/4/11 13:27'

import numpy as np
import pandas as pd


class gen_data:
    def __init__(self, data="D题附件1.xlsx", label="D题附件2.xlsx"):
        self.data_name = data
        self.label_name = label

    def read_data(self):
        # 获取整个数据集
        self.data = pd.read_excel("D题附件1.xlsx")
        self.label = pd.read_excel("D题附件2.xlsx")
        self.label = self.label.fillna(0)
        self.data = self.data[["转炉终点C", '转炉终点Mn', '钢水净重', '连铸正样C', '连铸正样Mn',
                               "钒铁(FeV50-A)", "钒铁(FeV50-B)", "硅铝合金FeAl30Si25", "硅铝锰合金球",
                               "硅锰面（硅锰渣）", "硅铁(合格块)", "硅铁FeSi75-B", "石油焦增碳剂",
                               "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)", "碳化硅(55%)", "硅钙碳脱氧剂", "转炉终点温度"]]
        # 251行之后转炉终点Mn为空
        self.data = self.data[:251]
        self.data_com = self.data.iloc[:, 5:]

    def cal_C(self):
        # C的历史回收率
        data_c = self.data_com.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        C = self.label.iloc[:, 1]
        components = (C != 0)
        components = np.array(self.label.iloc[:, 1][components])
        div_num = data_c * components
        sum_data = div_num.sum(axis=1)
        self.data["C收得率"] = (self.data['连铸正样C'] - self.data["转炉终点C"]) * self.data['钢水净重'] / sum_data

    def cal_Mn(self):
        # # Mn的历史回收率
        data_Mn = self.data_com.iloc[:, [3, 4, 8, 9]]
        Mn = self.label.iloc[:, 2]
        components = (Mn != 0)
        components = np.array(self.label.iloc[:, 2][components])
        div_num = data_Mn * components
        sum_data = div_num.sum(axis=1)
        self.data["Mn收得率"] = (self.data['连铸正样Mn'] - self.data["转炉终点Mn"]) * self.data['钢水净重'] / sum_data

    def run(self):
        self.read_data()
        self.cal_C()
        self.cal_Mn()
        # 对结果进行处理 去除inf和null值
        self.data.fillna(np.inf, inplace=True)
        self.data = self.data[~self.data["Mn收得率"].isin(['inf'])]
        self.data = self.data[~self.data["C收得率"].isin(['inf'])]

        # 将收得率大于1的进行剔除
        self.data = self.data[self.data["C收得率"] < 1]
        # 将转炉终点温度为0的剔除
        self.data = self.data[self.data["转炉终点温度"] != 0]
        # 储存结果文件
        self.data.to_excel("result.xlsx", index=False)

if __name__ == '__main__':
    item = gen_data()
    item.run()