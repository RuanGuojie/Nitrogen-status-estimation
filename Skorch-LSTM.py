#  -*- coding = utf-8 -*- 
#  @time2022/4/1511:02
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

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical, Continuous
from sklearn_genetic.plots import plot_fitness_evolution, plot_search_space
from sklearn_genetic.callbacks import LogbookSaver, ProgressBar
# See Regression Metrics to evaluate on test dataset
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
# 读取数据
dataset = pd.read_csv('N/AGB.csv')  # 更改数据
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
# Standard Normalization Preprocess
# normalize data to 0 mean and unit std
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

import skorch
from skorch import NeuralNetRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(42)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Set up Neural Network (lstm)
class LstmNetwork(nn.Module):
    def __init__(self, n_features=17, n_lstm_hidden=20, n_out=1, num_layers=1, bi=False):
        super().__init__()
        self.n_features = n_features  # 注意更改特征数量
        self.n_lstm_hidden = n_lstm_hidden  # 非常重要的参数，隐藏神经元不够的话预测值会饱和
        self.n_fc_hidden = n_lstm_hidden
        self.n_out = n_out
        self.num_layers = num_layers
        self.bi = bi
        # self.dropout = dropout
        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.n_lstm_hidden,
            num_layers=self.num_layers,
            bidirectional=self.bi,
            # dropout=self.dropout
        )

        self.fc = nn.Linear(
            in_features=n_lstm_hidden,
            out_features=n_out
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# Wrap Pytorch Neural Network in Skorch Wrapper
net = NeuralNetRegressor(
    module=LstmNetwork,
    criterion=nn.MSELoss,
    max_epochs=500,
    optimizer=optim.Adam,
    optimizer__lr=0.1,  # 学习率的影响也很大
)

# net.fit(X_train_scaled, y_train)
# y_pred_net = net.predict(X_test_scaled.astype(np.float32))
# print(r2_score(y_test, y_pred_net))

# genetic algorithm Hyperparameter Search
cv = KFold(n_splits=5, shuffle=True, random_state=42)

params = {
    # 'optimizer__lr': sp_randint(0.001, 0.1),
    # 'max_epochs':Integer(200, 300),
    # 'optimizer__lr': [0.001, 0.01, 0.1],
    # 'module__dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
    # 'module__n_lstm_hidden': [5, 10, 15, 20, 30],
    # 'module__num_layers': [1, 2, 3],
}
'''
gs = RandomizedSearchCV(net, params,
                        refit=True,
                        cv=cv,
                        scoring='neg_mean_squared_error',
                        n_iter=50,   # Number of parameter settings that are sampled
                        n_jobs=-1,
                        random_state=42)
'''

'''
# 进化算法优化超参数，速度太慢！
#  At least two parameters are advised to be provided in order to successfully make an optimization routine.
params = {
    'optimizer__lr': Continuous(0.001, 0.1),
    # 'max_epochs':Integer(200, 300),
    'module__dropout':Continuous(0.1, 0.5),
    'module__n_lstm_hidden':Integer(10, 30)
    }
gs = GASearchCV(
    estimator=net,
    cv=cv,
    scoring="neg_mean_squared_error",  # neg_mean_squared_error越大越好（越接近于0）
    population_size=15,
    generations=20,
    tournament_size=3,
    elitism=True,
    keep_top_k=4,
    crossover_probability=0.8,
    mutation_probability=0.1,
    param_grid=params,
    criteria="max",  # max if a higher scoring metric is better, min otherwise
    algorithm="eaMuCommaLambda",
    n_jobs=-1)
'''
gs = net.fit(X_train_scaled, y_train)
#gs.fit(X_train_scaled, y_train)
#print(gs.best_params_)
# print("Best k solutions: ", gs.hof)

'''
# Utility function to report best scores (found online)
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
# review top 10 results and parameters associated
report(gs.cv_results_,10)
'''

'''
# Display Learning curves to see if overfitting or underfitting data
# get training and validation loss
epochs = [i for i in range(len(gs.history))]
train_loss = gs.history[:, 'train_loss']
valid_loss = gs.history[:, 'valid_loss']
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
joblib.dump(gs, 'save/AGB.pkl')  # 更改文件名
# restore
regressor = joblib.load('save/AGB.pkl')

# predict on test data
y_pred = regressor.predict(X_test_scaled.astype(np.float32))
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
print(r2_score(y_test,y_pred))
plt.plot(y_pred, y_test, 'g*')
plt.xlabel('Predicted')
plt.ylabel('Observed')
# plt.title('$R^{2}$ visual')
plt.show()
# show where the big errors were
'''
errors = np.where(abs(y_test-y_pred)>.2)
for tup in zip(y_test[errors],y_pred[errors]):
    print(tup)
plot = plot_fitness_evolution(gs, metric="fitness")
plt.show()
plot_search_space(gs)
plt.show()
'''