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
        self.item = ["钢号", "转炉终点温度", "转炉终点C", '转炉终点Mn', '转炉终点S', '转炉终点P', '转炉终点Si', '钢水净重',
                     '连铸正样C', '连铸正样Mn', '连铸正样S', '连铸正样P', '连铸正样Si', "低铝硅铁",
                     "钒铁(FeV50-A)", "钒铁(FeV50-B)", "硅铝钙",
                     "硅铝合金FeAl30Si25", "硅铝锰合金球",
                     "硅锰面（硅锰渣）", "硅铁(合格块)", "硅铁FeSi75-B", "石油焦增碳剂",
                     "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)", "碳化硅(55%)", "硅钙碳脱氧剂"]
        self.data = self.data[self.item]
        # 251行之后转炉终点Mn为空

    def cal_C(self):
        # C的历史回收率
        # 810行之后转炉终点C为空
        self.left_data = self.data[810:]
        self.data = self.data[:810]
        self.data_com = self.data.iloc[:, 13:]
        data_c = self.data_com.iloc[:, [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13]]
        C = self.label.iloc[:, 1]
        components = (C != 0)
        components = np.array(self.label.iloc[:, 1][components])
        div_num = data_c * components
        sum_data = div_num.sum(axis=1)
        self.data["C收得率"] = (self.data['连铸正样C'] - self.data["转炉终点C"]) * self.data['钢水净重'] / sum_data
        self.left_data["C收得率"] = 0

    def cal_Mn(self):
        # # Mn的历史回收率
        # 251行之后转炉终点Mn为空
        data = self.data[:251]
        data_com = data.iloc[:, 13:]
        data_Mn = data_com.iloc[:, [5, 6, 10, 11]]
        Mn = self.label.iloc[:, 2]
        components = (Mn != 0)
        components = np.array(self.label.iloc[:, 2][components])
        div_num = data_Mn * components
        sum_data = div_num.sum(axis=1)
        # self.data["Mn元素的质量总和"] = sum_data
        self.data["Mn收得率"] = (self.data['连铸正样Mn'] - self.data["转炉终点Mn"]) * self.data['钢水净重'] / sum_data
        self.left_data["Mn收得率"] = 0

    def cal_S(self):
        data = self.data[:251]
        data_com = data.iloc[:, 13:]
        data_S = data_com.iloc[:, [1, 2, 6, 7, 8, 10, 11,12]]
        S = self.label.iloc[:, 3]
        components = (S != 0)
        components = np.array(self.label.iloc[:, 2][components])
        div_num = data_S * components
        sum_data = div_num.sum(axis=1)
        # self.data["Mn元素的质量总和"] = sum_data
        self.data["S收得率"] = (self.data['连铸正样S'] - self.data["转炉终点S"]) * self.data['钢水净重'] / sum_data
        self.left_data["S收得率"] = 0

    def cal_P(self):
        self.data = self.data[:810]
        self.data_com = self.data.iloc[:, 13:]
        data_P = self.data_com.iloc[:, [1, 2, 6, 7, 8, 10, 11]]
        P = self.label.iloc[:, 4]
        components = (P != 0)
        components = np.array(self.label.iloc[:, 1][components])
        div_num = data_P * components
        sum_data = div_num.sum(axis=1)
        # self.data["C元素的质量总和"] = sum_data
        self.data["P收得率"] = (self.data['连铸正样P'] - self.data["转炉终点P"]) * self.data['钢水净重'] / sum_data
        self.left_data["P收得率"] = 0

    def cal_Si(self):
        self.data = self.data[:810]
        self.data_com = self.data.iloc[:, 13:]
        data_Si = self.data_com.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]]
        Si = self.label.iloc[:, 5]
        components = (Si != 0)
        components = np.array(self.label.iloc[:, 1][components])
        div_num = data_Si * components
        sum_data = div_num.sum(axis=1)
        # self.data["C元素的质量总和"] = sum_data
        self.data["Si收得率"] = (self.data['连铸正样Si'] - self.data["转炉终点Si"]) * self.data['钢水净重'] / sum_data
        self.left_data["Si收得率"] = 0

    def run(self):
        self.read_data()
        self.cal_C()
        self.cal_Mn()
        self.cal_S()
        self.cal_P()
        self.cal_Si()
        item = ["Mn收得率","C收得率","S收得率","P收得率","Si收得率"]
        # 对结果进行处理 去除inf和null值
        self.data.fillna(np.inf, inplace=True)
        self.data.replace(np.inf,0,inplace=True)
        self.left_data.replace(np.inf, 0)
        # print(self.data["C收得率"])
        for i in item:
            self.data = self.data[~self.data[i].isin(['inf'])]
        # print(self.data["C收得率"])
        self.data = pd.concat([self.data, self.left_data], axis=0)
        # 对【Mn收得率","C收得率","S收得率","P收得率","Si收得率】进行数据的预处理
        # for i in item:
        #     self.data = self.data[(self.data[i] < 1) ]
        # 将转炉终点温度为0的剔除
        self.data = self.data[self.data["转炉终点温度"] != 0]
        self.data = self.data[~self.data["转炉终点温度"].isnull()]
        self.data = self.data[~self.data["钢水净重"].isnull()]
        for i in self.item:
            self.data = self.data[~(self.data[i] == np.inf)]
            self.data = self.data[~self.data[i].isnull()]
            self.data = self.data[~(self.data[i] == np.nan)]
        # 储存结果文件
        self.data.to_excel("process_C_Mn_Si_S_P_data.xlsx", index=False)


if __name__ == '__main__':
    item = gen_data()
    item.run()
