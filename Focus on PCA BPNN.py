import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# 数据加载和预处理
def load_and_preprocess_data():
    # 加载数据
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    # 分离特征和目标
    X = train_data.drop(['id', 'FloodProbability'], axis=1)
    y = train_data['FloodProbability']
    X_test = test_data.drop('id', axis=1)

    # 计算统计特征
    feature_median = X.median(axis=1).values.reshape(-1, 1)
    feature_min = X.min(axis=1).values.reshape(-1, 1)
    test_median = X_test.median(axis=1).values.reshape(-1, 1)
    test_min = X_test.min(axis=1).values.reshape(-1, 1)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    train_median, val_median = train_test_split(feature_median, test_size=0.2, random_state=42)
    train_min, val_min = train_test_split(feature_min, test_size=0.2, random_state=42)

    return (X_train, X_val, y_train, y_val, X_test_scaled,
            train_median, val_median, train_min, val_min,
            test_median, test_min)

def apply_pca(X_train, X_val, X_test, explained_variance_threshold=0.95):
    pca = PCA(n_components=explained_variance_threshold)  # 使用解释方差比作为标准
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_val_pca, X_test_pca, pca
# PCA降维



# BP神经网络
class BPNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(BPNeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


# 数据集类
class FloodDataset(Dataset):
    def __init__(self, X_pca, median, min_val, y=None):
        # 合并PCA特征和统计特征
        self.X = np.hstack([X_pca, median, min_val])
        self.X = torch.FloatTensor(self.X)
        # 将pandas Series转换为numpy数组
        self.y = torch.FloatTensor(y.values) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# 训练函数
def train_model(X_train_pca, train_median, train_min, y_train,
                X_val_pca, val_median, val_min, y_val,
                epochs=150, batch_size=64):
    # 创建数据加载器
    train_dataset = FloodDataset(X_train_pca, train_median, train_min, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 计算输入维度（PCA特征 + 中位数 + 最小值）
    input_size = X_train_pca.shape[1] + 2

    # 初始化模型
    model = BPNeuralNetwork(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    train_losses = []
    val_losses = []

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

        # 验证
        model.eval()
        with torch.no_grad():
            val_X = torch.FloatTensor(np.hstack([X_val_pca, val_median, val_min]))
            val_pred = model(val_X)
            val_loss = criterion(val_pred.squeeze(), torch.FloatTensor(y_val))

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss.item())

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}')

    # 绘制学习曲线
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model


# 评估函数
def evaluate_model(model, X_val_pca, val_median, val_min, y_val):
    model.eval()
    with torch.no_grad():
        val_X = torch.FloatTensor(np.hstack([X_val_pca, val_median, val_min]))
        val_pred = model(val_X).numpy().squeeze()

    # 计算评估指标
    mse = np.mean((y_val - val_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((y_val - val_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)

    # 绘制预测vs实际值散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, val_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.title('Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.show()

    return {'MSE': mse, 'RMSE': rmse, 'R2': r2}


# 主程序
def main():
    # 1. 加载和预处理数据
    print("Loading and preprocessing data...")
    (X_train, X_val, y_train, y_val, X_test,
     train_median, val_median, train_min, val_min,
     test_median, test_min) = load_and_preprocess_data()

    # 2. PCA降维
    print("\nApplying PCA...")
    X_train_pca, X_val_pca, X_test_pca, pca = apply_pca(X_train, X_val, X_test)
    print(f"Explained variance ratio sum: {np.sum(pca.explained_variance_ratio_):.4f}")

    # 3. 训练模型
    print("\nTraining model...")
    model = train_model(X_train_pca, train_median, train_min, y_train,
                        X_val_pca, val_median, val_min, y_val)

    # 4. 评估模型
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_val_pca, val_median, val_min, y_val)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # 5. 生成测试集预测
    print("\nGenerating test predictions...")
    model.eval()
    with torch.no_grad():
        test_X = torch.FloatTensor(np.hstack([X_test_pca, test_median, test_min]))
        test_pred = model(test_X).numpy().squeeze()

    # 保存预测结果
    test_predictions = pd.DataFrame({
        'id': range(len(test_pred)),
        'FloodProbability': test_pred
    })
    test_predictions.to_csv('bp_predictions.csv', index=False)
    print("Predictions saved to 'bp_predictions.csv'")


if __name__ == "__main__":
    main()