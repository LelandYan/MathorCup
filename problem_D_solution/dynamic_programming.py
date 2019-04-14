# _*_ coding: utf-8 _*_
__author__ = 'Yxp'
__date__ = '2019/4/14 16:22'

import numpy as np
import pandas as pd

data = pd.read_excel("result_C_Mn_Si_S_P_.xlsx")
each_money = pd.read_excel("D题附件2.xlsx",index_col=False)
item = ['低铝硅铁','钒铁(FeV50-A)','钒铁(FeV50-B)','硅铝钙','硅铝合金FeAl30Si25','硅铝锰合金球',
         '硅锰面（硅锰渣）','硅铁(合格块)','硅铁FeSi75-B','石油焦增碳剂','锰硅合金FeMn64Si27(合格块)',
         '锰硅合金FeMn68Si18(合格块)','碳化硅(55%)','硅钙碳脱氧剂']
mon = np.array([6500,205000,205000,11800,1000,8500,7600,6000,6000,4600,8150,8150,8150,8150]).reshape(-1,1)

item_mon = zip(item,mon)
for i in np.arange(data.shape[0]):
        data.loc[i,"成本"] = np.array(data.loc[i,item]).reshape(1,-1).dot(mon)[0][0]/1000
item_id = ["HRB400B","HRB400D","HRB500D","20MnKA","HRB500B","20MnKB","Q345B","Q235A","Q235"]
print(data)
data.to_excel("result_C_Mn_Si_S_P_last.xlsx")
