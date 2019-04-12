# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/12 7:38'

import pandas as pd
import numpy as np

data = pd.read_excel("D题附件1.xlsx")
data = data[["转炉终点C", '转炉终点Mn', '钢水净重', '连铸正样C', '连铸正样Mn',
                               "钒铁(FeV50-A)", "钒铁(FeV50-B)", "硅铝合金FeAl30Si25", "硅铝锰合金球",
                               "硅锰面（硅锰渣）", "硅铁(合格块)", "硅铁FeSi75-B", "石油焦增碳剂",
                               "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)", "碳化硅(55%)", "硅钙碳脱氧剂", "转炉终点温度"]]
data.to_excel("need_data.xlsx", index=False)