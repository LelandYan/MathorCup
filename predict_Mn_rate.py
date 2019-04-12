# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/12 15:46'
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
data = pd.read_excel("result.xlsx")
data = data.iloc[192:, :]
label = ["转炉终点温度", "转炉终点C", '钢水净重', '连铸正样C', '连铸正样Mn',
         "钒铁(FeV50-A)", "钒铁(FeV50-B)", "硅铝合金FeAl30Si25", "硅铝锰合金球",
         "硅锰面（硅锰渣）", "硅铁(合格块)", "硅铁FeSi75-B", "石油焦增碳剂",
         "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)", "碳化硅(55%)", "硅钙碳脱氧剂", "C收得率", "Mn收得率"]
Mn = ["转炉终点温度", "钢水净重", "钒铁(FeV50-B)", "硅锰面（硅锰渣）",
      "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)", "Mn收得率"]
data_input = data.loc[:, Mn[:-1]]
# print(data_input.shape)
clf = joblib.load("Mn.m")
print(data_input)
res = clf.predict(data_input)
print(res)
# data["Mn收得率"] = res
