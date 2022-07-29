#  -*- coding = utf-8 -*- 
#  @time 2022/4/15 10:46
#  Author:Ruanguojie


import warnings
warnings.filterwarnings('ignore')
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
# 画图模块
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# pip install sklearn-genetic-opt (not sklearn-genetic)
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical, Continuous
from sklearn_genetic.plots import plot_fitness_evolution, plot_search_space
from sklearn_genetic.callbacks import LogbookSaver, ProgressBar
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score

# 读取数据
dataset = pd.read_csv('N/AGB.csv')
# 打乱数据集：
dataset = dataset.sample(frac=1.0, random_state=42)
data = dataset.reset_index(drop=True)
# Dataset Analysis and Visualization
# display first 5 rows of dataset to see what we are looking at
print(dataset.head())
# show the distributions of data
print(dataset.describe())

target = dataset.pop('AGB')  # 标记放在最后一行

# Split into train/test datasets
# split data into train test
X_train, X_test, y_train, y_test = train_test_split(dataset.values.astype(np.float32),
                                                    target.values.reshape(-1, 1).astype(np.float32),
                                                    test_size=.2, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
# y_test_save = pd.DataFrame(y_test)
# y_test_save.to_csv('save/y_test_NNI.csv')
# Standard Normalization Preprocess
# normalize data to 0 mean and unit std
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

from skorch import NeuralNetRegressor
from sklearn.model_selection import RandomizedSearchCV
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.manual_seed(42)  # 固定Mymodule的随机种子
# Set up Neural Network (Multi-layer perceptron)
class MyModule(nn.Module):
    def __init__(self, input_size=17, num_units=10, nonlin=F.relu):
        super(MyModule, self).__init__()

        self.module = nn.Sequential(  # 快速搭建法Sequential
            nn.Linear(input_size, num_units),
            nn.ReLU(),
            nn.Linear(num_units,num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units,num_units),
            nn.ReLU(),  # 激励函数
            # nn.Dropout(p=drop), #加dropout层会降低性能
            nn.Linear(num_units, 1),
        )

    def forward(self, X):
        X = self.module(X)
        return X

'''
# Wrap Pytorch Neural Network in Skorch Wrapper
net = NeuralNetRegressor(
    MyModule,
    criterion=nn.MSELoss,  # 回归问题常用MSEloss
    max_epochs=500,  # 训练次数
    optimizer=optim.Adam,  # 优化器, Adam是个非常快的优化器
    optimizer__lr=.01,  # 学习率
)
'''

# genetic algorithm Hyperparameter Search
cv = KFold(n_splits=5, shuffle=True, random_state=42)
#  At least two parameters are advised to be provided in order to successfully make an optimization routine.
params = {
    'optimizer__lr': Continuous(0.001, 0.1),
    # 'max_epochs': Integer(200, 300),
    'module__num_units': Integer(5, 30),
    # 'module__drop': Continuous(0, 0.5),
}

'''
gs = GASearchCV(
    estimator=net,
    cv=cv,
    scoring="neg_mean_squared_error",  # neg_mean_squared_error越大越好（越接近于0）
    population_size=10,
    generations=40,  # 默认值为40
    tournament_size=3,
    elitism=True,
    keep_top_k=5,  # 默认值为1
    crossover_probability=0.8,
    mutation_probability=0.1,
    param_grid=params,
    criteria="max",  # max if a higher scoring metric is better, min otherwise
    algorithm="eaMuCommaLambda",
    n_jobs=-1)

gs.fit(X_train_scaled, y_train)
print(gs.best_params_)
print("Best k solutions: ", gs.hof)
plot = plot_fitness_evolution(gs, metric="fitness")
plt.show()
plot_search_space(gs)
plt.show()
'''

'''
# 随机搜索：速度比较快
params = {
    'optimizer__lr': (0.001, 0.01, 0.1),
    'max_epochs': (100, 200, 300),
    'module__num_units': (10, 20, 30, 40, 50),
    # 'module__drop': (0, 0.1, 0.2, 0.3, 0.4, 0.5),
}
gs = RandomizedSearchCV(net, params, refit=True, cv=cv, scoring='neg_mean_squared_error', n_iter=100)
gs.fit(X_train_scaled,y_train)
'''

'''
# Display Learning curves to see if overfitting or underfitting data
# get training and validation loss
epochs = [i for i in range(len(gs.best_estimator_.history))]
train_loss = gs.best_estimator_.history[:, 'train_loss']
valid_loss = gs.best_estimator_.history[:, 'valid_loss']
plt.plot(epochs, train_loss, 'g-')
plt.plot(epochs, valid_loss, 'r-')
plt.title('Training Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend(['Train', 'Validation'])
plt.show()
'''

# 保存模型
import joblib
# save
#joblib.dump(gs.best_estimator_, 'save/NNI-GA.pkl')  # 更改文件名
# restore
regressor = joblib.load('save/AGB.pkl')

# predict on test data
y_pred = regressor.predict(X_test_scaled.astype(np.float32))
# y_pred_save = pd.DataFrame(y_pred)
# y_pred_save.to_csv('save/y_pred_NNI.csv')
# get RMSE
RMSE = MSE(y_test, y_pred)**(1/2)
print(RMSE)
sns.kdeplot(y_pred.squeeze(), label='Predicted', shade=True)
sns.kdeplot(y_test.squeeze(), label='Observed', shade=True)
plt.xlabel('N_status')
plt.show()
sns.distplot(y_test.squeeze()-y_pred.squeeze(), label='error')
plt.xlabel('N_status Error')
plt.show()
# show R^2 plot
print(r2_score(y_test, y_pred))
plt.plot(y_pred, y_test, 'g*')
plt.xlabel('Predicted')
plt.ylabel('Observed')
# plt.title('$R^{2}$ visual')
plt.show()
# show where the big errors were
'''
errors = np.where(abs(y_test-y_pred) > .2)
for tup in zip(y_test[errors], y_pred[errors]):
    print(tup)
'''


