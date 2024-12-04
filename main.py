import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Input, Attention, Concatenate
from tensorflow.keras.utils import plot_model
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager  # 中文显示
from tqdm.keras import TqdmCallback  # 显示进度条
import tensorflow as tf

# 检查是否有可用的 GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置 TensorFlow 仅使用第一个 GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)

# 设置matplotlib的字体为SimHei，以便支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据加载与预处理
df = pd.read_csv("final_RossmannSales.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Store', 'Date'])

# 创建滞后特征和滑动平均特征
def feature_engineering(group):
    group['Sales_Lag1'] = group['Sales'].shift(1)
    group['Sales_Lag7'] = group['Sales'].shift(7)
    group['Sales_MA7'] = group['Sales'].rolling(window=7).mean()
    group['Sales_MA30'] = group['Sales'].rolling(window=30).mean()
    return group

df = df.groupby('Store').apply(feature_engineering)

# 对类别变量进行编码
df = pd.get_dummies(df, columns=['StoreType', 'Assortment', 'StateHoliday'], drop_first=True)

# 去掉滞后特征中生成的缺失值，并重设索引
df = df.dropna().reset_index(drop=True)

# 特征选择
features = ['Store', 'Customers', 'Promo', 'SchoolHoliday', 'Sales_Lag1', 'Sales_Lag7', 
            'Sales_MA7', 'Sales_MA30', 'DayOfWeek', 'Month', 'Season']
target = 'Sales'

# 划分训练集和测试集
train = df[df['Date'] < '2015-07-01']
test = df[df['Date'] >= '2015-07-01']

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# 数据标准化
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# XGBoost模型
def xgboost_model():
    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
    return model

# LSTM模型
def lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# CNN-LSTM模型
def cnn_lstm_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, 2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# CNN-LSTM-Attention模型
def cnn_lstm_attention_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 2, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    x = LSTM(32, activation='relu', return_sequences=True)(x)
    # 添加 Attention 层
    attention = Attention()([x, x])
    x = Concatenate()([x, attention])
    x = LSTM(32, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_model(model_func, X_train, y_train, X_test, y_test, model_type):
    # 为LSTM, CNN-LSTM, CNN-LSTM-Attention调整输入形状
    if model_type in ['lstm', 'cnn_lstm', 'cnn_lstm_attention']:
        X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    else:
        X_train_reshaped = X_train
        X_test_reshaped = X_test

    if model_type == 'xgboost':
        model = model_func()
        model.fit(X_train_reshaped, y_train.ravel())
    else:
        model = model_func(X_train_reshaped.shape[1:])
        model.fit(X_train_reshaped, y_train, epochs=1, batch_size=32, verbose=1, callbacks=[TqdmCallback(verbose=1)])

    y_pred = model.predict(X_test_reshaped)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    return rmse, mae, y_test, y_pred, model

# 进行一次实验并收集性能指标
results = []

xgboost_rmse, xgboost_mae, xgboost_y_true, xgboost_y_pred, xgboost_model_trained = evaluate_model(xgboost_model, X_train_scaled, y_train_scaled.ravel(), X_test_scaled, y_test_scaled.ravel(), 'xgboost')
lstm_rmse, lstm_mae, lstm_y_true, lstm_y_pred, lstm_model_trained = evaluate_model(lstm_model, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, 'lstm')
cnn_lstm_rmse, cnn_lstm_mae, cnn_lstm_y_true, cnn_lstm_y_pred, cnn_lstm_model_trained = evaluate_model(cnn_lstm_model, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, 'cnn_lstm')
cnn_lstm_attention_rmse, cnn_lstm_attention_mae, cnn_lstm_attention_y_true, cnn_lstm_attention_y_pred, cnn_lstm_attention_model_trained = evaluate_model(cnn_lstm_attention_model, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, 'cnn_lstm_attention')

results.append({
    'Model': 'XGBoost', 'RMSE': xgboost_rmse, 'MAE': xgboost_mae
})
results.append({
    'Model': 'LSTM', 'RMSE': lstm_rmse, 'MAE': lstm_mae
})
results.append({
    'Model': 'CNN-LSTM', 'RMSE': cnn_lstm_rmse, 'MAE': cnn_lstm_mae
})
results.append({
    'Model': 'CNN-LSTM-Attention', 'RMSE': cnn_lstm_attention_rmse, 'MAE': cnn_lstm_attention_mae
})

# 将结果转换为DataFrame
results_df = pd.DataFrame(results)

# 汇总表
print(results_df)

# 可视化结果
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='RMSE', data=results_df, ci="sd", capsize=0.1)
plt.title("模型性能 - RMSE", fontproperties='SimHei')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MAE', data=results_df, ci="sd", capsize=0.1)
plt.title("模型性能 - MAE", fontproperties='SimHei')
plt.show()

# 提取一个特定店铺的数据进行预测
store_id = 1  # 选择店铺ID为1的店铺
df_store = df[df['Store'] == store_id]

# 划分训练集和测试集
train_store = df_store[df_store['Date'] < '2015-07-01']
test_store = df_store[df_store['Date'] >= '2015-07-01']

X_train_store, y_train_store = train_store[features], train_store[target]
X_test_store, y_test_store = test_store[features], test_store[target]

# 数据标准化
X_train_store_scaled = scaler_X.transform(X_train_store)
X_test_store_scaled = scaler_X.transform(X_test_store)

y_train_store_scaled = scaler_y.transform(y_train_store.values.reshape(-1, 1))
y_test_store_scaled = scaler_y.transform(y_test_store.values.reshape(-1, 1))

# 转换为LSTM和CNN输入形状
X_train_store_lstm = np.reshape(X_train_store_scaled, (X_train_store_scaled.shape[0], X_train_store_scaled.shape[1], 1))
X_test_store_lstm = np.reshape(X_test_store_scaled, (X_test_store_scaled.shape[0], X_test_store_scaled.shape[1], 1))

# 使用训练好的模型进行预测
y_pred_store_xgboost_scaled = xgboost_model_trained.predict(X_test_store_scaled)
y_pred_store_lstm_scaled = lstm_model_trained.predict(X_test_store_lstm)
y_pred_store_cnn_lstm_scaled = cnn_lstm_model_trained.predict(X_test_store_lstm)
y_pred_store_cnn_lstm_attention_scaled = cnn_lstm_attention_model_trained.predict(X_test_store_lstm)

# 反标准化预测值和真实值
y_pred_store_xgboost = scaler_y.inverse_transform(y_pred_store_xgboost_scaled.reshape(-1, 1))
y_pred_store_lstm = scaler_y.inverse_transform(y_pred_store_lstm_scaled)
y_pred_store_cnn_lstm = scaler_y.inverse_transform(y_pred_store_cnn_lstm_scaled)
y_pred_store_cnn_lstm_attention = scaler_y.inverse_transform(y_pred_store_cnn_lstm_attention_scaled)
y_test_store = scaler_y.inverse_transform(y_test_store_scaled)

# 绘制拟合图
def plot_predictions(y_true, y_pred, model_name, dates):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label='真实值', color='blue', marker='o')
    plt.plot(dates, y_pred, label='预测值', color='red', linestyle='--', marker='x')
    plt.title(f"{model_name} 预测值与真实值拟合图", fontproperties='SimHei')
    plt.xlabel('日期', fontproperties='SimHei')
    plt.ylabel('销售额', fontproperties='SimHei')
    plt.legend()
    plt.show()

# 获取日期
dates_store = test_store['Date'].values

plot_predictions(y_test_store, y_pred_store_xgboost, "XGBoost", dates_store)
plot_predictions(y_test_store, y_pred_store_lstm, "LSTM", dates_store)
plot_predictions(y_test_store, y_pred_store_cnn_lstm, "CNN-LSTM", dates_store)
plot_predictions(y_test_store, y_pred_store_cnn_lstm_attention, "CNN-LSTM-Attention", dates_store)