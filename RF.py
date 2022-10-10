#  -*- coding = utf-8 -*- 
#  @time2022/4/1511:21
#  Author:Ruanguojie

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# figure module
import matplotlib.pyplot as plt
import seaborn as sns
import palettable
import shap  # model interpretability
import matplotlib

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
# read data
dataset = pd.read_csv('N/AGB.csv')
# randomly shuffled data
dataset = dataset.sample(frac=1.0, random_state=42)
data = dataset.reset_index(drop=True)
# Dataset Analysis and Visualization
# display first 5 rows of dataset to see what we are looking at
print(dataset.head())
# show the distributions of data
print(dataset.describe())

target = dataset.pop('AGB')

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
# X_train_scaled_save = pd.DataFrame(X_train_scaled)
# X_train_scaled_save.to_csv('X_train_scaled.csv')

# grid search for RF hyper parameter optimization
param_grid_RF = [{
    'n_estimators': [10, 50, 100, 500, 1000],
    'max_depth': [5, 10, 30, 50, 100],
    'min_samples_split':[2, 5, 10, 15],
    'min_samples_leaf':[1, 5, 10]
    }]
RF = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_RF, cv=5, n_jobs=-1)
RF.fit(X_train_scaled, np.ravel(y_train))
final_model_RF = RF.best_estimator_
print(RF.best_params_)

# save model
import joblib
# save
joblib.dump(final_model_RF, 'save2/AGB.pkl')
# restore
regressor = joblib.load('save2/AGB.pkl')

y_pred_RF = regressor.predict(X_test_scaled)
# y_pred_RF_save = pd.DataFrame(y_pred_RF)
# y_pred_RF_save.to_csv('save/y_predict_NNI.csv')
RF_train_score = regressor.score(X_train_scaled, np.ravel(y_train))
RF_test_score = r2_score(np.ravel(y_test), y_pred_RF)
RF_test_MSE = MSE(np.ravel(y_test), y_pred_RF)
RMSE_test = RF_test_MSE ** 0.5

print("r^2 of RF regression on training data: %.4f" % RF_train_score)
print("r^2 of RF regression on test data: %.4f" % RF_test_score)
print("RMSE of RF regression on test data: %.4f" % RMSE_test)

# plot the predicted and observed values using test data
'''
# scatter plot: style 1
# ax = plt.axes()
# plt.style.use('ggplot')  # figure style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # let the normal display of Chinese and negative sign
plt.figure(figsize=(3.54, 2.36), dpi=500)  # figure size and dpi
ax = sns.regplot(x=y_test, y=y_pred_RF, ci=95, scatter_kws=dict(linewidth=0.7, edgecolor='white'))
# plt.scatter(y_test, y_pred_RF, alpha=0.4)
# alpha:transparency，c:color，s:size，marker:shape of marker
plt.plot([0, y_test.max()],
         [0, y_test.max()],  # make the range of coordinate scale consistent
         '--',
         linewidth=2,
         c='pink')

# plt.title('NNI-RF', fontsize=15)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)  # hidden the lines on the top and right
# ax.spines['left'].set_position(('outward', 5))
# ax.spines['bottom'].set_position(('outward', 5))  # move axis
plt.xlabel('AGB_Measured', fontsize=10)
plt.ylabel('AGB_Predicted', fontsize=10)  # title of x/y-axis and font size
plt.title('AGB', fontsize=10)  # title of figure and font size
plt.tight_layout()
plt.subplots_adjust(left=0.20)
plt.savefig('AGB.jpeg')
plt.show()
'''

'''
# hexbin plot: style 2
plt.figure(figsize=(3.54, 2.36), dpi=500)  # figure size and dpi
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # let the normal display of Chinese and negative sign
ax = plt.axes()
TK = plt.gca()
TK.spines['bottom'].set_linewidth(0.5)  # control the lines of frame
TK.spines['left'].set_linewidth(0.5)
TK.spines['top'].set_linewidth(0.5)
TK.spines['right'].set_linewidth(0.5)
# fontcn = {'family': 'SimHei'}  # Chinese typeface
# fonten = {'family': 'Times New Roman'}  # English typeface

# match color using palettable
cmap = palettable.colorbrewer.sequential.Blues_9.mpl_colormap
plt.hexbin(
    np.ravel(y_test),
    y_pred_RF,
    gridsize=20,
    mincnt=1,  # important!
    cmap=cmap,
    edgecolors='white',
    linewidths=0)  # the bigger grid size, the more hexagon
plt.xlim(0, y_test.max())  # xlim
plt.ylim(0, y_test.max())  # ylim, must the same with xlim
plt.locator_params(nbins=5)  # the number of scales
plt.tick_params(labelsize=10, width=0.5)  # display of x/y-axis
plt.grid(ls='--', lw=0.5)  # add gird
ax.set_axisbelow(True)  # place grid at the bottom
ax.plot((0, 1), (0, 1), transform=ax.transAxes,
        ls='-', c='gray', linewidth=0.5)  
        # add 1:1 line, control line style, color, and line width
        
cb = plt.colorbar()  # add legend colorbar
# cb.set_label('Number of scatters',fontsize=10)  # legend label
cb.ax.tick_params(labelsize=10, width=0.5)  # label size of legend axis
cb.outline.set_linewidth(0.5)  # legend frame
# cb.set_ticks(np.linspace(1, 10, 10))

# titles
plt.xlabel('NNI_Measured', fontsize=10)  
plt.ylabel('NNI_Predicted', fontsize=10)  
#plt.title('NNI', fontsize=10) 
'''
# add a text box
''''
text = plt.text(
    x=0.5,   # remember to change
    y=1.5,
    s='R$\mathregular{^2}$ = %.4f' %
    RF_test_score +
    '\n' +
    'RMSE = %.4f' %   # remember to add units of RMSE
    RMSE_test,
    fontsize=8,
    bbox={
        'facecolor': 'white',
        'edgecolor': 'white',
        'alpha': 0.5,
        'pad': 0.5})
'''
'''
plt.tight_layout()
plt.subplots_adjust(left=0.20)
plt.savefig('NNI.jpeg')
plt.show()
'''


'''
# Model interpretability based on SHAP (SHapley Additive exPlanations)
# it can be used not only for tree-based models (RF here), but also for deep learning model

# summary_plot
# plt.figure(dpi=500)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

# use shap_values or shap_interaction_values
#explainer = shap.Explainer(regressor)
#shap_values = explainer(X_train_scaled)
explainer2 = shap.TreeExplainer(regressor)
shap_interaction_values = explainer2.shap_interaction_values(X_train_scaled)
shap.summary_plot(shap_interaction_values, X_train_scaled, show=False, plot_size=[3.44, 2.56],feature_names=dataset.columns,
                  color_bar=False, cmap='RdBu', alpha=0.9)
# plot_type='layered_violin' 
'''


'''
# dependence_plot
explainer3 = shap.TreeExplainer(regressor)
shap_values = explainer3.shap_interaction_values(X_train_scaled)
# print(shap_values)
# shap_values_save = pd.DataFrame(shap_values.reshape(2443*17,17))
# shap_values_save.to_csv('shap_value_NNI.csv')  # save interaction values to excel

shap.dependence_plot(('AGDD','RESAVI'),shap_values, X_train_scaled, feature_names=dataset.columns, show=False,
                     cmap='RdBu',
                     dot_size=10, alpha=0.8) 
'''

'''
fig = plt.gcf()  # gcf means "get current figure"
ax = plt.gca()  # gca means "get current axes"
#fig.set_figwidth(3.34)
#fig.set_figheight(2.56)

cb = plt.colorbar()  # redisplay colorbar
cb.set_label('Feature Value', fontsize=8)  
cb.ax.tick_params(labelsize=8, width=0.2)  
cb.outline.set_linewidth(0) 

#ax.set_xlim(-6000,6000) 
#ax.set_xlabel('SHAP Value', fontsize=8) 
# ax.set_ylabel('SHAP Value for AGDD', fontsize=8)
#ax.tick_params(labelsize=8) 
# ax.tick_params(axis='y', pad=-10)  # default yticklabel position is offset too far
#ax.spines['right'].set_visible(True)
#ax.spines['left'].set_visible(True)
#ax.spines['top'].set_visible(True)  
#plt.title('AGB-SHAP', fontsize=8)  
plt.tight_layout()
# plt.subplots_adjust(right=0.96)
plt.savefig('AGB-SHAP.jpeg', dpi=500)
plt.show()
'''

