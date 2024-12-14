import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据
original_data = pd.read_csv('flood.csv')
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 数据预处理
train_data_drop_id = train_data.drop(columns='id')
test_data_drop_id = test_data.drop(columns='id')

# 准备训练数据
X_train = train_data_drop_id.drop('FloodProbability', axis=1)
y_train = train_data_drop_id['FloodProbability']
X_test = test_data_drop_id

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)



warnings.filterwarnings('ignore')

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 创建模型保存目录
def create_model_directory():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = f'model_checkpoints_{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


# 数据加载和预处理
def load_and_preprocess_data(data_path):
    """加载和预处理数据"""
    data = pd.read_csv(data_path)

    # 分离特征和标签
    X = data.drop(['id', 'FloodProbability'], axis=1)
    y = data['FloodProbability']

    # 划分训练集、验证集和测试集
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (X_train_scaled, y_train.values,
            X_val_scaled, y_val.values,
            X_test_scaled, y_test.values,
            scaler)


class FloodDataset(Dataset):
    def __init__(self, X, y=None, classification=False):
        self.X = torch.FloatTensor(X)
        if y is not None:
            if classification:
                y = (y * 100).astype(int)
            self.y = torch.FloatTensor(y) if not classification else torch.LongTensor(y)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class ResNetRegression(nn.Module):
    def __init__(self, input_size):
        super(ResNetRegression, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.res_blocks = nn.ModuleList([
            self._make_res_block(128, 128),
            self._make_res_block(128, 64),
            self._make_res_block(64, 32)
        ])

        self.fc_final = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def _make_res_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout(out)

        identity = out
        for block in self.res_blocks:
            out = block(out)
            if out.shape == identity.shape:
                out += identity
            identity = out
            out = torch.relu(out)

        out = self.fc_final(out)
        return out


def save_model(model, model_path, scaler=None, metrics=None):
    """保存模型、缩放器和指标"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_architecture': type(model).__name__,
        'input_size': model.fc1.in_features,
        'scaler': scaler,
        'metrics': metrics
    }
    torch.save(checkpoint, model_path)


def load_model(model_path, device):
    """加载模型"""
    checkpoint = torch.load(model_path, map_location=device)
    input_size = checkpoint['input_size']

    model = ResNetRegression(input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    return model, checkpoint['scaler'], checkpoint['metrics']


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device, model_dir, patience=10):
    """训练模型"""
    model = model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_r2': []}

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()

                val_predictions.extend(outputs.squeeze().cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
        val_r2 = r2_score(val_targets, val_predictions)

        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Val Loss: {avg_val_loss:.4f}')
            print(f'Val RMSE: {val_rmse:.4f}')
            print(f'Val R2: {val_r2:.4f}')

        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            save_model(model,
                       f'{model_dir}/best_model.pth',
                       metrics={'val_rmse': val_rmse, 'val_r2': val_r2})
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping!")
            break

        # 定期保存检查点
        if (epoch + 1) % 50 == 0:
            save_model(model,
                       f'{model_dir}/model_epoch_{epoch + 1}.pth',
                       metrics={'val_rmse': val_rmse, 'val_r2': val_r2})

    return model, history


def evaluate_model(model, data_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            total_loss += loss.item()

            predictions.extend(outputs.squeeze().cpu().numpy())
            targets.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)

    return predictions, avg_loss, rmse, r2


def plot_training_history(history):
    """绘制训练历史"""
    plt.figure(figsize=(15, 5))

    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # RMSE曲线
    plt.subplot(1, 3, 2)
    plt.plot(history['val_rmse'])
    plt.title('Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')

    # R2曲线
    plt.subplot(1, 3, 3)
    plt.plot(history['val_r2'])
    plt.title('Validation R2 Score')
    plt.xlabel('Epoch')
    plt.ylabel('R2')

    plt.tight_layout()
    plt.show()


def plot_predictions(y_true, y_pred, title):
    """绘制预测vs实际值散点图"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.title(title)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.tight_layout()
    plt.show()


def main():
    # 创建模型保存目录
    model_dir = create_model_directory()

    # 加载和预处理数据
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler = \
        load_and_preprocess_data('train.csv')

    # 创建数据加载器
    train_dataset = FloodDataset(X_train_scaled, y_train)
    val_dataset = FloodDataset(X_val_scaled, y_val)
    test_dataset = FloodDataset(X_test_scaled, y_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 初始化模型和训练参数
    input_size = X_train_scaled.shape[1]
    model = ResNetRegression(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 200

    # 训练模型
    print("开始训练模型...")
    model, history = train_model(model, train_loader, val_loader, criterion,
                                 optimizer, num_epochs, device, model_dir)

    # 绘制训练历史
    plot_training_history(history)

    # 加载最佳模型进行测试
    best_model, _, best_metrics = load_model(f'{model_dir}/best_model.pth', device)
    print("\n最佳模型验证集性能:")
    print(f"RMSE: {best_metrics['val_rmse']:.4f}")
    print(f"R2 Score: {best_metrics['val_r2']:.4f}")

    # 在测试集上评估
    print("\n在测试集上评估模型...")
    test_predictions, test_loss, test_rmse, test_r2 = evaluate_model(
        best_model, test_loader, criterion, device
    )

    print("\n测试集性能:")
    print(f"Loss: {test_loss:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"R2 Score: {test_r2:.4f}")

    # 绘制测试集预测结果
    plot_predictions(y_test, test_predictions, 'Test Set: Actual vs Predicted')

    # 保存测试集预测结果
    test_results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': test_predictions,
        'Absolute_Error': np.abs(y_test - test_predictions)
    })
    test_results.to_csv(f'{model_dir}/test_results.csv', index=False)


if __name__ == "__main__":
    main()