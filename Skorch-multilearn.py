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
# figure module
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical, Continuous
from sklearn_genetic.plots import plot_fitness_evolution, plot_search_space
from sklearn_genetic.callbacks import LogbookSaver, ProgressBar
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score

# read data
dataset = pd.read_csv('N/mtl.csv')
# randomly shuffled data
dataset = dataset.sample(frac=1.0, random_state=42)
data = dataset.reset_index(drop=True)
# Dataset Analysis and Visualization
# display first 5 rows of dataset to see what we are looking at
print(dataset.head())
# show the distributions of data
print(dataset.describe())

target1 = dataset.pop('AGB')
target2 = dataset.pop('PNC')
target3 = dataset.pop('PNU')
target4 = dataset.pop('NNI')

# Split into train/test datasets
# split data into train test
X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test,  y4_train, y4_test = train_test_split(
    dataset.values.astype(np.float32),
    target1.values.reshape(-1, 1).astype(np.float32),
    target2.values.reshape(-1, 1).astype(np.float32),
    target3.values.reshape(-1, 1).astype(np.float32),
    target4.values.reshape(-1, 1).astype(np.float32),
    test_size=.2, random_state=42)

print(X_train.shape)
print(y1_train.shape)
print(y2_train.shape)
print(y3_train.shape)
print(y4_train.shape)
print(X_test.shape)

# re-stack
y_train = np.hstack((y1_train, y2_train, y3_train, y4_train))
y_test = np.hstack((y1_test, y2_test, y3_test, y4_test))
print(y_train.shape)
print(y_test.shape)

# Standard Normalization Preprocess
# normalize data to 0 mean and unit std
scaler1 = StandardScaler()
scaler1.fit(X_train)
X_train_scaled = scaler1.transform(X_train)
X_test_scaled = scaler1.transform(X_test)
scaler2 = StandardScaler()
scaler2.fit(y_train)   # Y must be standardized in multi-task learning, for eliminating the effect of dimensionality
y_train = scaler2.transform(y_train)
y_test = scaler2.transform(y_test)
mean_y1_test = np.mean(y1_test)
mean_y2_test = np.mean(y2_test)
mean_y3_test = np.mean(y3_test)
mean_y4_test = np.mean(y4_test)

print(mean_y1_test)
print(mean_y2_test)
print(mean_y3_test)
print(mean_y4_test)

from skorch import NeuralNetRegressor
from sklearn.model_selection import RandomizedSearchCV
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(42)
# Set up Neural Network (Multi-layer perceptron)
class MultiTaskRegressorModule(nn.Module):
    def __init__(self, input_size=17, num_units=20, output_size=4):
        super(MultiTaskRegressorModule, self).__init__()

        self.module = nn.Sequential(
            nn.Linear(input_size, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            # nn.Dropout(p=drop),
            nn.Linear(num_units, output_size),
        )

    def forward(self, X):
        X = self.module(X)
        return X

# set MultiOutputMSELoss
class MultiOutputMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, input, target):
        n = 4  # number of targets
        assert y_train.shape[1] == n
        losses = [nn.functional.mse_loss(input[:, i], target[:, i]) for i in range(n)]
        losses_weighted = [self.weights[i] * losses[i] for i in range(n)]
        return sum(losses_weighted)


# Wrap Pytorch Neural Network in Skorch Wrapper
net = NeuralNetRegressor(
    MultiTaskRegressorModule,
    criterion=MultiOutputMSELoss([10, 10, 10, 10]),
    #  We can change it to nn.MSEloss, or change weights of different targets manually
    max_epochs=1000,
    optimizer=optim.Adam,
    optimizer__lr=.01,  # learning rate
)
net.fit(X_train_scaled, y_train)

# optimization results of five times
# first: 10, 6, 6, 10
# second: 10, 10, 10, 10
# third: 10, 10, 10, 10  0.11798073
# fourth: 9, 9, 10, 10  0.10353291
# fifth: 10, 10, 10, 9 0.10067127

'''
# optimize weights of MultiOutputMSELoss using genetic algorithm (not change manually) in scikit-opt
# Get more information at:
# https://scikit-opt.github.io/scikit-opt/#/zh/README
# https://scikit-opt.github.io/scikit-opt/#/en/README
# Genetic Algorithm, Particle Swarm Optimization, Simulated Annealing, Ant Colony Optimization Algorithm,
# Immune Algorithm, Artificial Fish Swarm Algorithm, Differential Evolution and TSP(Traveling salesman)

# Step1：define your problem
def schaffer(weights):
    x1, x2, x3, x4 = weights
    net = NeuralNetRegressor(
        MultiTaskRegressorModule,
        criterion=MultiOutputMSELoss([x1, x2, x3, x4]), 
        max_epochs=100,  
        optimizer=optim.Adam,  
        optimizer__lr=.01,
    )
    net.fit(X_train_scaled,y_train)
    y_pred = net.predict(X_train_scaled)
    result = MSE(y_train, y_pred)  # optimization for small
    return result

# Step2: do Genetic Algorithm (or others)
from sko.GA import GA
from sko.PSO import PSO
from sko.tools import set_run_mode
set_run_mode(schaffer, 'parallel')  # accelerate

# take GA as an example
ga = GA(func=schaffer, n_dim=4, size_pop=50, max_iter=50, prob_mut=0.001, lb=[1, 1, 1, 1], ub=[10, 10, 10, 10], precision=1)
# precision=1  round up the variable
# best_x, best_y = ga.run()
ga.run()
print('best_x:', ga.best_x, '\n', 'best_y:', ga.best_y)

# Step3: Plot the result:
Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()
'''

# take PSO as an example
'''
pso = PSO(func=schaffer, n_dim=4, pop=40, max_iter=150, lb=[1, 1, 1, 1], ub=[10, 10, 10, 10], w=0.8, c1=0.5, c2=0.5)
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
plt.plot(pso.gbest_y_hist)
plt.show()
'''

# save model
import joblib
# save
joblib.dump(net, 'save/mtl-ga.pkl')
# restore
regressor = joblib.load('save/mtl-ga.pkl')

'''
# Save model
save_filename = 'save/NNI.pth'
torch.save(mtl, save_filename)
print('Saved as %s' % save_filename)

# Start evaluating model
model.eval()
'''

# predict on test data
DNN_train_score = regressor.score(X_train_scaled, y_train)
y_train_pred = regressor.predict(X_train_scaled)
y_train_mse = MSE(y_train,y_train_pred)
y_pred_DNN = regressor.predict(X_test_scaled.astype(np.float32))
y_test = scaler2.inverse_transform(y_test)  # dis-standardized
y_pred_DNN = scaler2.inverse_transform(y_pred_DNN)
DNN_test_score = r2_score(y_test, y_pred_DNN)
DNN_test_score_AGB = r2_score(y_test[:, 0], y_pred_DNN[:, 0])
DNN_test_score_PNC = r2_score(y_test[:, 1], y_pred_DNN[:, 1])
DNN_test_score_PNU = r2_score(y_test[:, 2], y_pred_DNN[:, 2])
DNN_test_score_NNI = r2_score(y_test[:, 3], y_pred_DNN[:, 3])
DNN_test_MSE_AGB = MSE(y_test[:, 0], y_pred_DNN[:, 0])
DNN_test_MSE_PNC = MSE(y_test[:, 1], y_pred_DNN[:, 1])
DNN_test_MSE_PNU = MSE(y_test[:, 2], y_pred_DNN[:, 2])
DNN_test_MSE_NNI = MSE(y_test[:, 3], y_pred_DNN[:, 3])
RMSE_test_AGB = DNN_test_MSE_AGB ** 0.5
RMSE_test_PNC = DNN_test_MSE_PNC ** 0.5
RMSE_test_PNU = DNN_test_MSE_PNU ** 0.5
RMSE_test_NNI = DNN_test_MSE_NNI ** 0.5
rRMSE_AGB = RMSE_test_AGB/mean_y1_test
rRMSE_PNC = RMSE_test_PNC/mean_y2_test
rRMSE_PNU = RMSE_test_PNU/mean_y3_test
rRMSE_NNI = RMSE_test_NNI/mean_y4_test

print("r^2 of DNN on training data: %.4f" % DNN_train_score)
print("MSE of DNN on training data: %.4f" % y_train_mse)  # not dis-standardized
print("r^2 of DNN on test data: %.4f" % DNN_test_score)  # average R2 values of four tasks

# show the results of different tasks
print("r^2 of DNN on test data(AGB): %.4f" % DNN_test_score_AGB)
print("r^2 of DNN on test data(PNC): %.4f" % DNN_test_score_PNC)
print("r^2 of DNN on test data(PNU): %.4f" % DNN_test_score_PNU)
print("r^2 of DNN on test data(NNI): %.4f" % DNN_test_score_NNI)
print("RMSE of DNN on test data(AGB): %.4f" % RMSE_test_AGB)
print("RMSE of DNN on test data(PNC): %.4f" % RMSE_test_PNC)
print("RMSE of DNN on test data(PNU): %.4f" % RMSE_test_PNU)
print("RMSE of DNN on test data(NNI): %.4f" % RMSE_test_NNI)
print("rRMSE of DNN on test data(AGB): %.4f" % rRMSE_AGB)
print("rRMSE of DNN on test data(PNC): %.4f" % rRMSE_PNC)
print("rRMSE of DNN on test data(PNU): %.4f" % rRMSE_PNU)
print("rRMSE of DNN on test data(NNI): %.4f" % rRMSE_NNI)

'''
# show R^2 plot
plt.plot(y_pred_DNN, y_test, 'g*')
plt.xlabel('Predicted')
plt.ylabel('Observed')
# plt.title('$R^{2}$ visual')
plt.show()
'''


