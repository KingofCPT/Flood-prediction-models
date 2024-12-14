# Flood-prediction-models
This project applyed multiple models and comparsion using Kaggle "flood prediction dataset. It Explores the Limitations of Traditional Methods and the Role of Statistical Features
Following Library should be installed:
lightgbm,catboost,scikit-learn(sklearn),torch,seaborn,matplotlib,pandas,numpy and other libraries required in the file. Installed the libraries you don't have.

No specific version required, installed the latest version of libraries are recommended.

将所有的程序文件与train.csv, test.csv, flood.csv 放在同一个文件夹下，如果数据集不在同一个文件夹中，需要使用绝对路径替换代码中导入文件处的相对路径。
运行代码，等待后图像和结果会逐一显示，大部分代码由于数据集庞大的原因运行时间很久，大概估计在3-6小时之间，少部分代码运行时间可能超过6小时，如果没有报错，请耐心等待，所有代码经过验证可以完整独立运行。
predict.ipynb 文件使用jupyter notebook 或 Pycharm 运行。用于直观的展示数据探索性分析的各个结果，所有的数据探索性结果都在这个文件中。后续其他文件主要有关模型训练。
Basic model.py 文件展现了五个基础模型的训练，评估和可视化： Linear Regression, Random Forest, XGBoost, BP Neural Network, MLP.模型训练完毕后会保存五个模型分别在test.csv上的训练结果

"Focus on PCA BPNN", "Focus on PCA MLP and linear", "Focus on PCA Tree Model"分别对比了降维前后BP神经网络，MLP，线性回归和多个树模型的性能变化。

"Focus on ResNet"展示了使用ResNet对洪水概率数据集训练的性能及其可视化。
"Parallel"与Basic model.py代码文件基本一致，区别在与个别神经网络和随机森林的参数不同，该代码文件用于保留副本，调参和改进模型。
"Focus on linear regression"用于使用特征统计量进行线性回归的学习，同时进行单特征线性回归来分析每个特征统计量是否与洪水概率有线性关系。
"Improving Tree Model"用于在树模型中引入统计量特征进行学习，提升树模型的效果。
"Ensemble learning"则是使用了集成学习尝试对模型进行改进，并可视化了集成学习模型的性能以及与其他基础模型的对比

"Performance Visualization"基于保存的模型性能与信息，可以直接快速的展示集成学习模型与基础模型的性能对比，并展示了集成学习在训练集和验证集上的性能指标，计算了他们的overfitting ratio(过拟合比）。

如果不想等待那么久的时间，可以打开“加载模型和数据集”这一文件，里面通过加载saved_models里面保存好的训练模型来直接展示面模型性能与结果。
Predicted File using Test Dataset里面保存了所有的模型预测文件，可以前往kaggle flood prediction月赛的提交渠道中提交文件夹中的文件夹中的文件，查看在test.csv上的官方准确度。
