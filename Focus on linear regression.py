import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import statsmodels.api as sm
import warnings

plt.style.use('seaborn-v0_8')

warnings.filterwarnings('ignore')
# 加载数据
original_data = pd.read_csv('flood.csv')
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 数据预处理
train_data_drop_id = train_data.drop(columns='id')
test_data_drop_id = test_data.drop(columns='id')

X = train_data_drop_id.drop('FloodProbability', axis=1)
y = train_data_drop_id['FloodProbability']
X_test = test_data_drop_id


def analyze_statistical_features(dataset, target):
    """分析统计特征的重要性"""
    # 创建统计特征
    features = dataset.columns.tolist()
    stats_df = pd.DataFrame()

    stats_df['mean'] = dataset[features].mean(axis=1)
    stats_df['std'] = dataset[features].std(axis=1)
    stats_df['max'] = dataset[features].max(axis=1)
    stats_df['min'] = dataset[features].min(axis=1)
    stats_df['median'] = dataset[features].median(axis=1)
    stats_df['skew'] = dataset[features].skew(axis=1)
    stats_df['kurtosis'] = dataset[features].kurtosis(axis=1)
    stats_df['range'] = stats_df['max'] - stats_df['min']
    stats_df['iqr'] = dataset[features].quantile(0.75, axis=1) - dataset[features].quantile(0.25, axis=1)

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(stats_df)
    X_scaled = pd.DataFrame(X_scaled, columns=stats_df.columns)

    # 使用statsmodels进行多元线性回归
    X_with_const = sm.add_constant(X_scaled)
    model = sm.OLS(target, X_with_const).fit()

    # 打印回归结果
    print("\n多元线性回归结果（统计特征整体）：")
    print(model.summary())

    # 可视化系数
    plt.figure(figsize=(12, 6))
    coef_df = pd.DataFrame({
        'Feature': stats_df.columns,
        'Coefficient': model.params[1:],
        'P-value': model.pvalues[1:]
    })
    coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=True)

    colors = ['red' if p < 0.05 else 'gray' for p in coef_df['P-value']]
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
    plt.title('Statistical Features Coefficients\n(Red: statistically significant, p<0.05)')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.show()

    return stats_df, coef_df


def analyze_individual_features(dataset, target):
    """分析单个统计特征与目标变量的关系并可视化"""
    features = dataset.columns.tolist()
    results = []

    for feature in features:
        # 简单线性回归
        X = dataset[feature].values.reshape(-1, 1)

        # 检查数据中是否有NaN
        if np.isnan(X).any() or np.isnan(target).any():
            continue

        model = LinearRegression()
        model.fit(X, target)
        y_pred = model.predict(X)
        r2 = r2_score(target, y_pred)

        # 使用statsmodels获取详细统计信息
        X_with_const = sm.add_constant(X)
        detailed_model = sm.OLS(target, X_with_const).fit()

        results.append({
            'Feature': feature,
            'Coefficient': model.coef_[0],
            'R2': r2,
            'P-value': detailed_model.pvalues[1]
        })

        # 为该特征单独绘制散点图和回归线
        plt.figure(figsize=(6, 4))
        sns.regplot(x=dataset[feature], y=target, color='blue', scatter_kws={'alpha': 0.5})
        plt.title(
            f'Single Feature Linear Regression\nFeature: {feature}\nR² = {r2:.3f}, Coef = {model.coef_[0]:.3f}, P-value={detailed_model.pvalues[1]:.3e}')
        plt.xlabel(feature)
        plt.ylabel('FloodProbability')
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results)


# 主分析流程
dataset = pd.concat([original_data, train_data.drop('id', axis=1)], ignore_index=True)
dataset = dataset.dropna(subset=["FloodProbability"])
dataset = dataset.dropna()

X_original = dataset.drop(["FloodProbability"], axis=1)
y = dataset["FloodProbability"]

# 分析统计特征
stats_df, stats_importance = analyze_statistical_features(X_original, y)

# 对统计特征单独进行分析并可视化
feature_importance_stats = analyze_individual_features(stats_df, y)

# 打印统计特征单独回归的特征重要性排名
print("\n单个统计特征重要性排名（根据R2）：")
print(feature_importance_stats.sort_values('R2', ascending=False))

# 综合可视化统计特征重要性
plt.figure(figsize=(12, 6))
feature_importance_stats = feature_importance_stats.sort_values('R2', ascending=True)
colors = ['red' if p < 0.05 else 'gray' for p in feature_importance_stats['P-value']]
plt.barh(feature_importance_stats['Feature'], feature_importance_stats['R2'], color=colors)
plt.title('Statistical Feature Importance (R²)\n(Red: statistically significant, p<0.05)')
plt.xlabel('R² Score')
plt.tight_layout()
plt.show()

# 相关性热图（统计特征与FloodProbability）
plt.figure(figsize=(12, 8))
correlation_matrix = pd.concat([stats_df, pd.Series(y, name='FloodProbability')], axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap (Statistical Features)')
plt.tight_layout()
plt.show()

# 打印统计分析结果
print("\n统计特征描述：")
print(stats_df.describe())

print("\n统计特征与目标变量的相关性：")
correlations = stats_df.apply(lambda x: stats.pearsonr(x, y)[0])
p_values = stats_df.apply(lambda x: stats.pearsonr(x, y)[1])
correlation_df = pd.DataFrame({
    'Correlation': correlations,
    'P-value': p_values
})
print(correlation_df.sort_values('Correlation', ascending=False))
