# 导入所需的库
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

# 创建模型保存目录
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# 设置样式
sns.set()
sns.set_palette('viridis')
SNS_CMAP = 'viridis'

# 加载数据
original_data = pd.read_csv('flood.csv')
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 数据预处理
train_data_drop_id = train_data.drop(columns='id')
test_data_drop_id = test_data.drop(columns='id')

# 准备训练数据和验证数据
X = train_data_drop_id.drop('FloodProbability', axis=1)
y = train_data_drop_id['FloodProbability']
X_test = test_data_drop_id

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 将标准化后的数据转换为DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
def add_statistical_features(X):
    # 计算每个样本的统计特征
    X_stats = pd.DataFrame()
    X_stats['feature_min'] = X.min(axis=1)
    X_stats['feature_median'] = X.median(axis=1)
    X_stats['feature_range'] = X.max(axis=1) - X.min(axis=1)
    X_stats['feature_mean'] = X.mean(axis=1)
    X_stats['feature_max'] = X.max(axis=1)

    # 合并原特征和统计特征
    X_enhanced = pd.concat([X, X_stats], axis=1)
    return X_enhanced


# 对训练集、验证集和测试集添加统计特征
X_train_enhanced = add_statistical_features(X_train)
X_val_enhanced = add_statistical_features(X_val)
X_test_enhanced = add_statistical_features(X_test)

# 对增强后的特征进行标准化
scaler_enhanced = StandardScaler()
X_train_scaled_enhanced = scaler_enhanced.fit_transform(X_train_enhanced)
X_val_scaled_enhanced = scaler_enhanced.transform(X_val_enhanced)
X_test_scaled_enhanced = scaler_enhanced.transform(X_test_enhanced)

# 转换为DataFrame以保持列名
X_train_scaled_enhanced = pd.DataFrame(X_train_scaled_enhanced, columns=X_train_enhanced.columns)
X_val_scaled_enhanced = pd.DataFrame(X_val_scaled_enhanced, columns=X_val_enhanced.columns)
X_test_scaled_enhanced = pd.DataFrame(X_test_scaled_enhanced, columns=X_test_enhanced.columns)

# 修改模型字典，只保留需要的模型
models_enhanced = {
    'Random Forest': RandomForestRegressor(n_estimators=80, random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'MLP': MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=1000, random_state=42)
}

# 训练和评估模型
results_enhanced = {
    'Model': [],
    'Training Time': [],
    'Train RMSE': [],
    'Train R2': [],
    'Val RMSE': [],
    'Val R2': [],
    'Cross Val Score': []
}

# 训练和评估增强特征的模型
for name, model in models_enhanced.items():
    print(f"\n训练带统计特征的 {name}...")

    start_time = time.time()

    model.fit(X_train_scaled_enhanced, y_train)
    train_pred = model.predict(X_train_scaled_enhanced)
    val_pred = model.predict(X_val_scaled_enhanced)

    # 保存模型
    joblib.dump(model, f'saved_models/{name.replace(" ", "_")}_with_stats.pkl')

    training_time = time.time() - start_time

    # 计算评估指标
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_r2 = r2_score(y_val, val_pred)

    results_enhanced['Model'].append(f"{name} with Stats")
    results_enhanced['Training Time'].append(training_time)
    results_enhanced['Train RMSE'].append(train_rmse)
    results_enhanced['Train R2'].append(train_r2)
    results_enhanced['Val RMSE'].append(val_rmse)
    results_enhanced['Val R2'].append(val_r2)

    cv_scores = cross_val_score(model, X_train_scaled_enhanced, y_train, cv=5)
    results_enhanced['Cross Val Score'].append(cv_scores.mean())

    # 生成测试集预测结果
    test_predictions = model.predict(X_test_scaled_enhanced)
    predictions_df = pd.DataFrame({
        'id': test_data['id'],
        'Predicted_FloodProbability': test_predictions
    })
    predictions_df.to_csv(f'{name.replace(" ", "_")}_predictions_considering_statistics.csv', index=False)

# 打印增强特征模型的详细结果
results_df_enhanced = pd.DataFrame(results_enhanced)
print("\n带统计特征的模型性能比较:")
print(results_df_enhanced.to_string(index=False))