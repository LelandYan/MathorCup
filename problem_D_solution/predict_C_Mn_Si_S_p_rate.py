# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/13 22:55'

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor


class cal_accuracy:
    def __init__(self, name="process_C_Mn_Si_S_P_data.xlsx"):
        self.file_name = name

    def read_data(self, filename=None):
        if filename:
            self.data = pd.read_excel(filename)
        else:
            self.data = pd.read_excel(self.file_name)

    def C_Mn_S_P_Si_model_cal_C_Mn_S_P_Si(self):
        data = self.data[(self.data["C收得率"] >= 0) | (self.data["C收得率"] < 1)]
        label = data.iloc[:193, -5:-1]
        self.label = ["转炉终点温度", "转炉终点C", "转炉终点S", "转炉终点Si", '钢水净重', "钒铁(FeV50-A)", "钒铁(FeV50-B)", "硅铝合金FeAl30Si25",
                      "硅铝锰合金球", "硅锰面（硅锰渣）", "硅铁(合格块)", "硅铁FeSi75-B", "石油焦增碳剂",
                      "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)", "碳化硅(55%)", "硅钙碳脱氧剂", ]
        # 剔除, '转炉终点Mn'
        data = data.loc[:192, self.label]
        return data, label

    def C_Mn_S_P_Si_model_cal_Mn_S_P(self):
        data = self.data[(self.data["Mn收得率"] >= 0) | (self.data["Mn收得率"] < 1)]
        Mn_label = data.iloc[:192, -4]
        self.label = ["转炉终点温度", "转炉终点C", '钢水净重', "钒铁(FeV50-A)", "钒铁(FeV50-B)", "硅铝合金FeAl30Si25", "硅铝锰合金球",
                      "硅锰面（硅锰渣）", "硅铁(合格块)", "硅铁FeSi75-B", "石油焦增碳剂",
                      "锰硅合金FeMn64Si27(合格块)", "锰硅合金FeMn68Si18(合格块)", "碳化硅(55%)", "硅钙碳脱氧剂", ]
        # 剔除, "连铸正样Mn"
        data_Mn = data.loc[:192, self.label]
        return data_Mn, Mn_label

    def train(self, C_Mn_S_P_Si_cal_Mn_S_P=False,
              C_Mn_S_P_Si_cal_C_Mn_S_P_Si=False):
        data = None
        label = None
        if C_Mn_S_P_Si_cal_Mn_S_P:
            data, label = self.C_Mn_S_P_Si_model_cal_Mn_S_P()
        if C_Mn_S_P_Si_cal_C_Mn_S_P_Si:
            data, label = self.C_Mn_S_P_Si_model_cal_C_Mn_S_P_Si()
        self.linear_model(data, label)
        # self.network_model(data,label)

    def network_model(self, data, label):
        try:
            output = label.shape[1]
        except:
            output = 1
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
        self.std = StandardScaler()
        self.std.fit(X_train)
        X_train = self.std.transform(X_train)
        X_test = self.std.transform(X_test)
        feature = X_train.shape[1]
        x = tf.placeholder(tf.float32, [None, feature], name='X')
        y = tf.placeholder(tf.float32, [None, output], name='Y')
        with tf.name_scope("Model1"):
            w = tf.Variable(tf.random_uniform([feature, output]), name="W")
            b = tf.Variable(1.0, name='b')

            def model(x, w, b):
                return tf.matmul(x, w) + b

            pred = model(x, w, b)
        train_epochs = 60
        learning_rate = 0.01
        loss_function = tf.reduce_mean(tf.pow(y - pred, 2))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        loss_list = []  # 用于保存loss值的列表
        for epoch in range(train_epochs):
            loss_sum = 0.0
            for xs, ys in zip(X_train, y_train):
                xs = xs.reshape(-1, feature)
                ys = ys.reshape(-1, output)
                # feed数据必须和Placeholder的shape一致
                _, loss = sess.run([optimizer, loss_function], feed_dict={x: xs, y: ys})

                loss_sum = loss_sum + loss

                loss_list.append(loss)  # 每步添加一次

            b0temp = b.eval(session=sess)  # 训练中当前变量b值
            w0temp = w.eval(session=sess)  # 训练中当前权重w值
            loss_average = loss_sum / len(y_train)  # 当前训练中的平均损失

            #     loss_list.append(loss_average)           #每轮添加一次
            print("epoch=", epoch + 1, "loss=", loss_average)

        predict = []
        for xs in X_test:
            xs = xs.reshape(-1, feature)
            if output == 1:
                predict.append(sess.run(pred, feed_dict={x: xs}).ravel()[0])
        predict = np.array(predict)
        # 模型的存储
        # joblib.dump(self.model, "Mn.m")
        print(np.sqrt(mean_squared_error(predict, y_test)))

    def linear_model(self, data, label):
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
        self.std = StandardScaler()
        self.std.fit(X_train)
        X_train = self.std.transform(X_train)
        X_test = self.std.transform(X_test)

        # self.model = RandomForestRegressor()  # 0.00178279872547 0.00130315829688
        self.model = LinearRegression()  # 8.36667068081e-05  0.00016716587287
        # self.model = Ridge() # 7.8575505755e-05              0.000232658506956
        # self.model = Lasso() # 0.0046516170403                  0.00238670854501
        # self.model = SVR()  # 0.00328161901572                 0.00307726323407
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # 模型的存储
        # joblib.dump(self.model, "Mn.m")
        print(np.sqrt(mean_squared_error(y_pred, y_test)))

    def run(self):
        self.read_data()
        self.train(C_Mn_S_P_Si_cal_Mn_S_P=True)
        self.fill_Mn_S_P()
        self.read_data(filename="result_C_Mn_Si_S_P_.xlsx")
        self.train(C_Mn_S_P_Si_cal_C_Mn_S_P_Si=True)
        self.fill_C_Mn_S_P_Si()

    def fill_Mn_S_P(self):
        data = self.data.iloc[229:610, :]
        data_input = data.loc[:, self.label]
        data_input = self.std.transform(data_input)
        res = np.array(self.model.predict(data_input))
        for i in np.arange(len(res)):
            self.data.loc[192 + i:, "Mn收得率"] = res[i][0]
            self.data.loc[192 + i:, "S收得率"] = res[i][1]
            self.data.loc[192 + i:, "P收得率"] = res[i][2]
        self.data.to_excel("result_C_Mn_Si_S_P_.xlsx", index=False)

    def fill_C_Mn_S_P_Si(self):
        data = self.data.iloc[610:, :]
        data_input = data.loc[:, self.label]
        data_input = self.std.transform(data_input)
        res = np.array(self.model.predict(data_input))
        for i in np.arange(len(res)):
            self.data.loc[610 + i:, "C收得率"] = res[i][0]
            self.data.loc[610 + i:, "Mn收得率"] = res[i][1]
            self.data.loc[610 + i:, "S收得率"] = res[i][2]
            self.data.loc[610 + i:, "P收得率"] = res[i][3]
            self.data.loc[610 + i:, "Si收得率"] = res[i][4]
        self.data.to_excel("result_C_Mn_Si_S_P_.xlsx", index=False)


if __name__ == '__main__':
    item = cal_accuracy("process_C_Mn_Si_S_P_data.xlsx")
    item.run()
