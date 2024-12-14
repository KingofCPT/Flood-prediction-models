import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_preprocess_data():
    # 加载数据
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    # 分离特征和目标
    X = train_data.drop(['id', 'FloodProbability'], axis=1)
    y = train_data['FloodProbability']
    X_test = test_data.drop('id', axis=1)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val, X_test_scaled


def apply_pca(X_train, X_val, X_test, explained_variance_threshold=0.95):
    pca = PCA(n_components=explained_variance_threshold)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    print(f"Original features: {X_train.shape[1]}")
    print(f"PCA features: {X_train_pca.shape[1]}")
    print(f"Explained variance ratio sum: {np.sum(pca.explained_variance_ratio_):.4f}")

    return X_train_pca, X_val_pca, X_test_pca, pca


def train_and_evaluate_models(X_train, X_val, y_train, y_val, model_type, is_pca=False):
    if model_type == "Linear":
        model = LinearRegression()
    else:  # MLP
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            max_iter=1000,
            random_state=42,
            early_stopping=True
        )

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # 计算指标
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'train_r2': r2_score(y_train, train_pred),
        'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
        'val_r2': r2_score(y_val, val_pred)
    }

    # 绘制预测vs实际值图
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_train, train_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    plt.title(f'{model_type} - Training Set {"(PCA)" if is_pca else ""}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    plt.subplot(1, 2, 2)
    plt.scatter(y_val, val_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.title(f'{model_type} - Validation Set {"(PCA)" if is_pca else ""}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    plt.tight_layout()
    plt.show()

    return model, metrics


def main():
    # 1. 加载和预处理数据
    print("Loading and preprocessing data...")
    X_train, X_val, y_train, y_val, X_test = load_and_preprocess_data()

    # 2. 训练原始特征模型
    print("\nTraining models with original features...")
    results = []

    for model_type in ["Linear", "MLP"]:
        print(f"\nTraining {model_type} Regression...")
        model, metrics = train_and_evaluate_models(X_train, X_val, y_train, y_val, model_type)
        results.append({
            'Model': f'{model_type} (Original)',
            **metrics
        })

    # 3. PCA降维
    print("\nApplying PCA...")
    X_train_pca, X_val_pca, X_test_pca, pca = apply_pca(X_train, X_val, X_test)

    # 4. 训练PCA特征模型
    print("\nTraining models with PCA features...")
    for model_type in ["Linear", "MLP"]:
        print(f"\nTraining {model_type} Regression with PCA...")
        model, metrics = train_and_evaluate_models(
            X_train_pca, X_val_pca, y_train, y_val, model_type, is_pca=True
        )
        results.append({
            'Model': f'{model_type} (PCA)',
            **metrics
        })

    # 5. 比较结果
    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df)

    # 6. 可视化比较
    metrics_to_plot = ['train_rmse', 'val_rmse', 'train_r2', 'val_r2']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for i, metric in enumerate(metrics_to_plot):
        sns.barplot(x='Model', y=metric, data=results_df, ax=axes[i])
        axes[i].set_title(metric.upper())
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # 7. 保存预测结果
    for model_type in ["Linear", "MLP"]:
        # 使用原始特征
        model_orig = train_and_evaluate_models(X_train, X_val, y_train, y_val, model_type)[0]
        pred_orig = model_orig.predict(X_test)
        pd.DataFrame({
            'id': range(len(pred_orig)),
            'FloodProbability': pred_orig
        }).to_csv(f'{model_type.lower()}_predictions_original.csv', index=False)

        # 使用PCA特征
        model_pca = train_and_evaluate_models(X_train_pca, X_val_pca, y_train, y_val, model_type)[0]
        pred_pca = model_pca.predict(X_test_pca)
        pd.DataFrame({
            'id': range(len(pred_pca)),
            'FloodProbability': pred_pca
        }).to_csv(f'{model_type.lower()}_predictions_pca.csv', index=False)


if __name__ == "__main__":
    main()


