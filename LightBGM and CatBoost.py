import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import os

# 创建保存模型的目录
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# 加载数据
print("Loading data...")
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 数据预处理
X = train_data.drop(columns=['id', 'FloodProbability'])
y = train_data['FloodProbability']
X_test = test_data.drop(columns=['id'])

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 将标准化后的数据转为DataFrame以保留特征名
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# 训练LightGBM
print("\nTraining LightGBM...")
lgbm = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    num_leaves=31,
    random_state=42
)
lgbm.fit(X_train_scaled, y_train)

# 训练CatBoost
print("\nTraining CatBoost...")
catboost = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=True
)
catboost.fit(X_train_scaled, y_train)


# 评估模型
def evaluate_model(model, name, X_train, X_val, y_train, y_val):
    # 训练集性能
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(((train_pred - y_train) ** 2).mean())

    # 验证集性能
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(((val_pred - y_val) ** 2).mean())

    print(f"\n{name} Performance:")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")

    return model.predict(X_test_scaled)


# 评估并保存预测结果
print("\nEvaluating models...")

# 评估LightGBM
lgbm_test_pred = evaluate_model(lgbm, "LightGBM", X_train_scaled, X_val_scaled, y_train, y_val)
pd.DataFrame({
    'id': test_data['id'],
    'Predicted_FloodProbability': lgbm_test_pred
}).to_csv('LightGBM_predictions.csv', index=False)

# 评估LightGBM
catboost_test_pred = evaluate_model(lgbm, "CatBoost", X_train_scaled, X_val_scaled, y_train, y_val)
pd.DataFrame({
    'id': test_data['id'],
    'Predicted_FloodProbability': lgbm_test_pred
}).to_csv('CatBoost_predictions.csv', index=False)

# 保存模型
print("\nSaving models...")
joblib.dump(lgbm, 'saved_models/LightGBM.pkl')

# 保存模型
print("\nSaving models...")
joblib.dump(lgbm, 'saved_models/CatBoost.pkl')
print("\nTraining and evaluation complete!")