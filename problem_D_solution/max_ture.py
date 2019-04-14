# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/14 21:44'

import numpy as np
import pandas as pd


HRB400B= pd.read_excel("HRB500B.xlsx")
HRB400B["最优化"] = HRB400B['成本'] + 1 / HRB400B["C收得率"]
# print(HRB400B.sort_values(HRB400B["最优化"]))
print(HRB400B["最优化"].sort_values(ascending=False))
# HRB400D= pd.read_excel("HRB400D.xlsx")
# HRB500D= pd.read_excel("HRB500D.xlsx")
# _20MnKA= pd.read_excel("20MnKA.xlsx")
# HRB500B= pd.read_excel("HRB500B.xlsx")
# _20MnKB= pd.read_excel("20MnKB.xlsx")
# Q345B= pd.read_excel("Q345B.xlsx")
# Q235A= pd.read_excel("Q235A.xlsx")
# Q235= pd.read_excel("Q235.xlsx")