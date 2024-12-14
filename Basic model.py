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


# 定义BP神经网络
class BPNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(BPNeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


# 定义数据集类
class FloodDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# BP神经网络训练函数
def train_bp_network(X_train, y_train, X_val, y_val):
    train_dataset = FloodDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = BPNeuralNetwork(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0025)

    train_losses = []
    val_losses = []
    epochs = 150

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 计算验证集损失
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val))
            val_loss = criterion(val_pred.squeeze(), torch.FloatTensor(y_val))

        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss.item())

        if epoch % 10 == 0:
            print(
                f'Epoch [{epoch}/{epochs}], Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss.item():.4f}')

    # 绘制学习曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model


def load_models():
    loaded_models = {}

    # 加载标准化器
    scaler = joblib.load('saved_models/scaler.pkl')

    # 加载BP神经网络
    bp_model = BPNeuralNetwork(X_train.shape[1])
    bp_model.load_state_dict(torch.load('saved_models/bp_model.pth'))
    bp_model.eval()
    loaded_models['BP Neural Network'] = bp_model

    # 加载其他机器学习模型
    model_names = ['Linear Regression', 'Random Forest', 'XGBoost', 'MLP']
    for name in model_names:
        model_path = f'saved_models/{name.replace(" ", "_")}.pkl'
        if os.path.exists(model_path):
            loaded_models[name] = joblib.load(model_path)

    return loaded_models, scaler


def predict_with_loaded_models(X_new):
    """
    使用保存的模型进行预测
    """
    loaded_models, scaler = load_models()
    predictions = {}

    # 数据标准化
    X_new_scaled = scaler.transform(X_new)

    for name, model in loaded_models.items():
        if name == 'BP Neural Network':
            X_tensor = torch.FloatTensor(X_new_scaled)
            with torch.no_grad():
                pred = model(X_tensor).numpy().squeeze()
        else:
            pred = model.predict(X_new_scaled)
        predictions[name] = pred

    return predictions


def evaluate_loaded_models(X, y):
    """
    评估加载的模型性能
    """
    loaded_models, scaler = load_models()
    X_scaled = scaler.transform(X)

    results = {
        'Model': [],
        'RMSE': [],
        'R2': []
    }

    for name, model in loaded_models.items():
        if name == 'BP Neural Network':
            X_tensor = torch.FloatTensor(X_scaled)
            with torch.no_grad():
                pred = model(X_tensor).numpy().squeeze()
        else:
            pred = model.predict(X_scaled)

        rmse = np.sqrt(mean_squared_error(y, pred))
        r2 = r2_score(y, pred)

        results['Model'].append(name)
        results['RMSE'].append(rmse)
        results['R2'].append(r2)

    return pd.DataFrame(results)


# 创建模型字典
models = {
    'Linear Regression': LinearRegression(),
    'BP Neural Network': None,
    'Random Forest': RandomForestRegressor(n_estimators=80, random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'MLP': MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=1000, random_state=42)
}

# 存储结果
results = {
    'Model': [],
    'Training Time': [],
    'Train RMSE': [],
    'Train R2': [],
    'Val RMSE': [],
    'Val R2': [],
    'Cross Val Score': []
}

# 训练和评估模型
for name, model in models.items():
    print(f"\n训练 {name}...")

    start_time = time.time()

    if name == 'BP Neural Network':
        bp_model = train_bp_network(X_train_scaled.values, y_train.values,
                                    X_val_scaled.values, y_val.values)
        train_pred = bp_model(torch.FloatTensor(X_train_scaled.values)).detach().numpy().squeeze()
        val_pred = bp_model(torch.FloatTensor(X_val_scaled.values)).detach().numpy().squeeze()

        # 保存BP神经网络模型
        torch.save(bp_model.state_dict(), f'saved_models/bp_model.pth')
        # 保存标准化器
        joblib.dump(scaler, 'saved_models/scaler.pkl')
    else:
        model.fit(X_train_scaled, y_train)
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)
        # 保存其他机器学习模型
        joblib.dump(model, f'saved_models/{name.replace(" ", "_")}.pkl')

    training_time = time.time() - start_time

    # 计算评估指标
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_r2 = r2_score(y_val, val_pred)

    results['Model'].append(name)
    results['Training Time'].append(training_time)
    results['Train RMSE'].append(train_rmse)
    results['Train R2'].append(train_r2)
    results['Val RMSE'].append(val_rmse)
    results['Val R2'].append(val_r2)

    if name != 'BP Neural Network':
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        results['Cross Val Score'].append(cv_scores.mean())
    else:
        results['Cross Val Score'].append(None)

# 创建结果DataFrame
results_df = pd.DataFrame(results)

# 可视化部分
# 1. 训练集和验证集的RMSE比较
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
data = pd.DataFrame({
    'Model': results['Model'] * 2,
    'RMSE': results['Train RMSE'] + results['Val RMSE'],
    'Dataset': ['Train'] * len(results['Model']) + ['Validation'] * len(results['Model'])
})
sns.barplot(x='Model', y='RMSE', hue='Dataset', data=data)
plt.xticks(rotation=45)
plt.title('Train vs Validation RMSE')

# 2. 训练集和验证集的R2比较
plt.subplot(1, 2, 2)
data = pd.DataFrame({
    'Model': results['Model'] * 2,
    'R2': results['Train R2'] + results['Val R2'],
    'Dataset': ['Train'] * len(results['Model']) + ['Validation'] * len(results['Model'])
})
sns.barplot(x='Model', y='R2', hue='Dataset', data=data)
plt.xticks(rotation=45)
plt.title('Train vs Validation R2')
plt.tight_layout()
plt.show()

# 3. 预测vs实际值比较
plt.figure(figsize=(15, 10))
for i, (name, model) in enumerate(models.items(), 1):
    plt.subplot(3, 2, i)
    if name == 'BP Neural Network':
        train_pred = bp_model(torch.FloatTensor(X_train_scaled.values)).detach().numpy().squeeze()
    else:
        train_pred = model.predict(X_train_scaled)
    plt.scatter(y_train, train_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    plt.title(f'{name}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
plt.tight_layout()
plt.show()

# 打印详细结果
print("\n详细模型性能比较:")
print(results_df.to_string(index=False))

# 生成test集预测结果
for name, model in models.items():
    if name == 'BP Neural Network':
        test_predictions = bp_model(torch.FloatTensor(X_test_scaled.values)).detach().numpy().squeeze()
    else:
        test_predictions = model.predict(X_test_scaled)
    predictions_df = pd.DataFrame({
        'id': test_data['id'],
        'Predicted_FloodProbability': test_predictions
    })
    predictions_df.to_csv(f'{name.replace(" ", "_")}_predictions.csv', index=False)

# 使用保存的模型进行预测示例
print("\n使用保存的模型进行预测和评估:")
new_predictions = predict_with_loaded_models(X_test)
performance_metrics = evaluate_loaded_models(X_val, y_val)
print("\n加载模型后的性能评估:")
print(performance_metrics)

