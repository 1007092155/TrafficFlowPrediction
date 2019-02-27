"""
Processing the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def process_data(train, test, lags):
    """Process data
    Reshape and split train\test data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
    # Returns
        x_train: ndarray.
        y_train: ndarray.
        x_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    """
    # 取车流量那一列的数据，并进行缺失值补全
    # df1 = pd.read_csv(train, encoding='utf-8', names=['week', 'timeSlot', 'holiday', 'rrr',
    #                                                   'Vt6', 'Vt5', 'Vt4', 'Vt3', 'Vt2', 'Vt1', 'Vt']).fillna(0)
    # df2 = pd.read_csv(test, encoding='utf-8', names=['week', 'timeSlot', 'holiday', 'rrr',
    #                                                  'Vt6', 'Vt5', 'Vt4', 'Vt3', 'Vt2', 'Vt1', 'Vt']).fillna(0)
    df1 = pd.read_csv(train, encoding='utf-8', names=['week', 'timeSlot', 'holiday', 'rrr', 'Vt']).fillna(0)
    df2 = pd.read_csv(test, encoding='utf-8', names=['week', 'timeSlot', 'holiday', 'rrr', 'Vt']).fillna(0)
    # scaler = StandardScaler().fit(df1[attr].values)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1.values[1:, :])
    scaler_y = MinMaxScaler(feature_range=(0, 1)).fit(df1.values[1:, -1].reshape(-1, 1))
    minmax_df1 = scaler.transform(df1.values[1:, :])
    minmax_df2 = scaler.transform(df2.values[1:, :])
    # 将训练数据和测试数据均处理为[n-lags,lags]的矩阵
    train, test = [], []
    for i in range(lags, minmax_df1.shape[0]):
        x = minmax_df1[i - lags: i, :-1].reshape(1, -1)
        y = minmax_df1[i - 1, -1].reshape(1, -1)
        train.append(np.hstack((x, y))[0])
    for i in range(lags, minmax_df2.shape[0]):
        x = minmax_df2[i - lags: i, :-1].reshape(1, -1)
        y = minmax_df2[i - 1, -1].reshape(1, -1)
        test.append(np.hstack((x, y))[0])

    train = np.array(train)
    test = np.array(test)
    # 打乱训练数据顺序
    np.random.shuffle(train)

    # 样本集和标签集X、y
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = test[:, :-1]
    y_test = test[:, -1]

    return x_train, y_train, x_test, y_test, scaler_y
