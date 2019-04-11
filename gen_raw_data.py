# _*_ coding: utf-8 _*_
__author__ = 'Yxp'
__date__ = '2019/4/11 13:27'

import numpy as np
import pandas as pd

# 获取整个数据集
data = pd.read_excel("D题附件1.xlsx")
label = pd.read_excel("D题附件2.xlsx")
label = label.fillna(0)
data = data[["转炉终点C", '转炉终点Mn', '钢水净重', '连铸正样C', '连铸正样Mn',
             "钒铁(FeV50-A)", "钒铁(FeV50-B)", "硅铝合金FeAl30Si25", "硅铝锰合金球",
             "硅锰面（硅锰渣）", "硅铁(合格块)", "硅铁FeSi75-B", "石油焦增碳剂",
             "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)", "碳化硅(55%)", "硅钙碳脱氧剂","转炉终点温度"]]

# 251行之后转炉终点Mn为空
data = data[:251]
data_com = data.iloc[:, 5:]

# C的历史回收率
data_c = data_com.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
C = label.iloc[:, 1]
components = (C != 0)
components = np.array(label.iloc[:, 1][components])
div_num = data_c * components
sum_data = div_num.sum(axis=1)
data["C收得率"] = (data['连铸正样C'] - data["转炉终点C"]) * data['钢水净重'] / sum_data
# # Mn的历史回收率
data_Mn = data_com.iloc[:, [3, 4, 8, 9]]
Mn = label.iloc[:, 2]
components = (Mn != 0)
components = np.array(label.iloc[:, 2][components])
div_num = data_Mn * components
sum_data = div_num.sum(axis=1)
data["Mn收得率"] = (data['连铸正样Mn'] - data["转炉终点Mn"]) * data['钢水净重'] / sum_data

# 对结果进行处理 去除inf和null值
data.fillna(np.inf, inplace=True)
data = data[~data["Mn收得率"].isin(['inf'])]
data = data[~data["C收得率"].isin(['inf'])]

# 将收得率大于1的进行剔除
data = data[data["C收得率"] < 1]
# 将转炉终点温度为0的剔除
data = data[data["转炉终点温度"] != 0]
# 储存结果文件
data.to_excel("result.xlsx", index=False)
