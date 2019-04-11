# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/11 13:27'

import numpy as np
import pandas as pd

data = pd.read_excel("history.xlsx")
data = data[["转炉终点C",'转炉终点Mn','钢水净重','连铸正样C','连铸正样Mn',"氮化钒铁FeV55N11-A",
             "低铝硅铁","钒氮合金(进口)","钒铁(FeV50-A)","钒铁(FeV50-B)","硅铝钙","硅铝合金FeAl30Si25"
            ,"硅铝锰合金球","硅锰面（硅锰渣）","硅铁(合格块)","硅铁FeSi75-B","石油焦增碳剂","锰硅合金FeMn64Si27(合格块)",
             "锰硅合金FeMn68Si18(合格块)","碳化硅(55%)","硅钙碳脱氧剂"]]

print(data[:250].info())