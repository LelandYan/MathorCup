# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/14 21:01'

import numpy as np
import pandas as pd

data = pd.read_excel("result_C_Mn_Si_S_P_last.xlsx")
item_id = ["HRB400B", "HRB400D", "HRB500D", "20MnKA", "HRB500B", "20MnKB", "Q345B", "Q235A", "Q235"]
HRB400B = data[data["钢号"].str.contains("HRB400B")]
HRB400D = data[data["钢号"].str.contains("HRB400D")]
HRB500D = data[data["钢号"].str.contains("HRB500D")]
_20MnKA = data[data["钢号"].str.contains("20MnKA")]
HRB500B = data[data["钢号"].str.contains("HRB500B")]
_20MnKB = data[data["钢号"].str.contains("20MnKB")]
Q345B = data[data["钢号"].str.contains("Q345B")]
Q235A = data[data["钢号"].str.contains("Q235A")]
Q235 = data[data["钢号"].str.contains("Q235")]


HRB400B.to_excel("HRB400B.xlsx")
HRB400D.to_excel("HRB400D.xlsx")
HRB500D.to_excel("HRB500D.xlsx")
_20MnKA.to_excel("20MnKA.xlsx")
HRB500B.to_excel("HRB500B.xlsx")
_20MnKB.to_excel("20MnKB.xlsx")
Q345B.to_excel("Q345B.xlsx")
Q235A.to_excel("Q235A.xlsx")
Q235.to_excel("Q235.xlsx")