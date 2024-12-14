import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置风格
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11

# 准备数据
# 创建性能比较的数据框
data = {
    'Model': ['LR', 'MLP', 'RF', 'XGBoost', 'BPNN', 'Average', 'Weighted', 'Stacking'],
    'RMSE': [0.0201, 0.0190, 0.0302, 0.0223, 0.0207, 0.0204, 0.0226, 0.0295],
    'R2': [0.8449, 0.8612, 0.6491, 0.8091, 0.8354, 0.8399, 0.8035, 0.6643],
    'Type': ['Individual']*5 + ['Ensemble']*3
}
df = pd.DataFrame(data)

# 创建一个图形，包含两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# RMSE比较图
sns.barplot(data=df, x='Model', y='RMSE', hue='Type',
            palette={'Individual': '#2878B5', 'Ensemble': '#C82423'},
            ax=ax1)
ax1.set_title('(a) RMSE Comparison', pad=15)
ax1.set_xlabel('Model')
ax1.set_ylabel('RMSE')

# 添加数值标签
for i in ax1.containers:
    ax1.bar_label(i, fmt='%.4f', padding=3)

# R2比较图
sns.barplot(data=df, x='Model', y='R2', hue='Type',
            palette={'Individual': '#2878B5', 'Ensemble': '#C82423'},
            ax=ax2)
ax2.set_title('(b) R² Comparison', pad=15)
ax2.set_xlabel('Model')
ax2.set_ylabel('R²')

# 添加数值标签
for i in ax2.containers:
    ax2.bar_label(i, fmt='%.4f', padding=3)

# 调整布局
plt.tight_layout()
plt.savefig('model_comparison_seaborn.png', dpi=300, bbox_inches='tight')
plt.show()

# 过拟合比较图
plt.figure(figsize=(10, 6))
overfitting_data = {
    'Ensemble Method': ['Average', 'Weighted', 'Stacking'],
    'Overfitting Ratio': [1.1853, 1.4466, 3.7044]
}
df_overfitting = pd.DataFrame(overfitting_data)

# 创建过拟合比较图
ax = sns.barplot(data=df_overfitting, x='Ensemble Method', y='Overfitting Ratio',
                 color='#2878B5')
plt.title('Overfitting Ratio Comparison of Ensemble Methods', pad=15)

# 添加数值标签
for i in ax.containers:
    ax.bar_label(i, fmt='%.4f', padding=3)

# 设置y轴范围，使图表更美观
plt.ylim(0, max(df_overfitting['Overfitting Ratio']) * 1.2)

plt.tight_layout()
plt.savefig('overfitting_comparison_seaborn.png', dpi=300, bbox_inches='tight')
plt.show()

# 创建训练集vs验证集性能对比图
ensemble_performance = {
    'Ensemble': ['Average', 'Weighted', 'Stacking'] * 2,
    'Set': ['Train'] * 3 + ['Validation'] * 3,
    'RMSE': [0.0172, 0.0156, 0.0080, 0.0204, 0.0226, 0.0295],
    'R2': [0.8863, 0.9063, 0.9756, 0.8399, 0.8035, 0.6643]
}
df_ensemble = pd.DataFrame(ensemble_performance)

# 创建训练集vs验证集性能对比图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# RMSE对比
sns.barplot(data=df_ensemble, x='Ensemble', y='RMSE', hue='Set',
            palette=['#2878B5', '#C82423'], ax=ax1)
ax1.set_title('(a) Train vs Validation RMSE', pad=15)
for i in ax1.containers:
    ax1.bar_label(i, fmt='%.4f', padding=3)

# R2对比
sns.barplot(data=df_ensemble, x='Ensemble', y='R2', hue='Set',
            palette=['#2878B5', '#C82423'], ax=ax2)
ax2.set_title('(b) Train vs Validation R²', pad=15)
for i in ax2.containers:
    ax2.bar_label(i, fmt='%.4f', padding=3)

plt.tight_layout()
plt.savefig('ensemble_train_val_comparison_seaborn.png', dpi=300, bbox_inches='tight')
plt.show()