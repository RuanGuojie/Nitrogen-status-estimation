#  -*- coding = utf-8 -*- 
#  @time2022/4/1511:21
#  Author:Ruanguojie

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# 画图模块
import matplotlib.pyplot as plt
import seaborn as sns
import palettable
import shap
import matplotlib

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
# 读取数据
dataset = pd.read_csv('N/NNI.csv')
# 打乱数据集：
dataset = dataset.sample(frac=1.0, random_state=42)
data = dataset.reset_index(drop=True)
# Dataset Analysis and Visualization
# display first 5 rows of dataset to see what we are looking at
print(dataset.head())
# show the distributions of data
print(dataset.describe())

target = dataset.pop('NNI')  # 标记放在最后一行

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

'''
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
'''
# 保存模型
import joblib
# save
# joblib.dump(final_model_RF, 'save/NNI.pkl')  # 更改文件名
# restore
regressor = joblib.load('save/NNI.pkl')

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

'''
# 散点图1
# ax = plt.axes()
# plt.style.use('ggplot')  # 绘图风格
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 中文编码与负号的正常显示
plt.figure(figsize=(3.54, 2.36), dpi=500)  # 画布大小(英寸）和分辨率
ax = sns.regplot(x=y_test, y=y_pred_RF, ci=95, scatter_kws=dict(linewidth=0.7, edgecolor='white'))
# plt.scatter(y_test, y_pred_RF, alpha=0.4)
# alpha控制透明度，c控制颜色，s控制大小，marker控制形状
plt.plot([0, y_test.max()],
         [0, y_test.max()],  # 横纵坐标刻度需要保持一致
         '--',
         linewidth=2,
         c='pink')

# plt.title('NNI-RF 实测值VS.预测值', fontsize=15)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)  # 隐藏顶部和右侧的实线
# ax.spines['left'].set_position(('outward', 5))
# ax.spines['bottom'].set_position(('outward', 5))  # 偏移axis
plt.xlabel('AGB_Measured', fontsize=10)
plt.ylabel('AGB_Predicted', fontsize=10)  # 横纵坐标轴标题
plt.title('AGB', fontsize=10)  # 图表标题
plt.tight_layout()
plt.subplots_adjust(left=0.20)
plt.savefig('AGB3.jpeg')
plt.show()
'''


# 散点图2
plt.figure(figsize=(3.54, 2.36), dpi=500)  # 画布大小和分辨率
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 中文编码与负号的正常显示
ax = plt.axes()
TK = plt.gca()
TK.spines['bottom'].set_linewidth(0.5)
TK.spines['left'].set_linewidth(0.5)
TK.spines['top'].set_linewidth(0.5)
TK.spines['right'].set_linewidth(0.5)
# fontcn = {'family': 'SimHei'}  # 中文字体
# fonten = {'family': 'Times New Roman'}  # 英文字体
# 使用palettable配色

cmap = palettable.colorbrewer.sequential.Blues_9.mpl_colormap
plt.hexbin(
    np.ravel(y_test),
    y_pred_RF,
    gridsize=20,
    mincnt=1,
    cmap=cmap,
    edgecolors='white',
    linewidths=0)  # gridsize越大，hex越多
plt.xlim(0, y_test.max())  # 横坐标刻度范围
plt.ylim(0, y_test.max())  # 纵坐标刻度范围, 两个刻度要一致，不然1：1线会错误！！
plt.locator_params(nbins=5)  # 坐标刻度数量
plt.tick_params(labelsize=10, width=0.5)  # 坐标轴刻度
plt.grid(ls='--', lw=0.5)  # 添加网格线
ax.set_axisbelow(True)  # 网格线置于底层
ax.plot((0, 1), (0, 1), transform=ax.transAxes,
        ls='-', c='gray', linewidth=0.5)  # 添加1：1线
cb = plt.colorbar()  # 添加图例colorbar
# cb.set_label('Number of scatters',fontsize=10)  # 图例标记
cb.ax.tick_params(labelsize=10, width=0.5)  # 设置图例刻度的字体大小
cb.outline.set_linewidth(0.5)  # 图例外框
# cb.set_ticks(np.linspace(1, 10, 10))
plt.xlabel('NNI_Measured', fontsize=10)  # 横坐标标题
plt.ylabel('NNI_Predicted', fontsize=10)  # 纵坐标轴标题
#plt.title('NNI', fontsize=10)  # 图表标题


# 添加文本框
''''
text = plt.text(
    x=0.5,   # 注意更改
    y=1.5,
    s='R$\mathregular{^2}$ = %.4f' %
    RF_test_score +
    '\n' +
    'RMSE = %.4f' %   # 加单位！！
    RMSE_test,
    fontsize=8,
    bbox={
        'facecolor': 'white',
        'edgecolor': 'white',
        'alpha': 0.5,
        'pad': 0.5})
'''


plt.tight_layout()
plt.subplots_adjust(left=0.20)
plt.savefig('NNI2.jpeg')
plt.show()


'''
# 特征重要性-基于SHAP值
# summary_plot
# plt.figure(dpi=500)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 中文编码与负号的正常显示

#explainer = shap.Explainer(regressor)
#shap_values = explainer(X_train_scaled)
explainer2 = shap.TreeExplainer(regressor)
shap_interaction_values = explainer2.shap_interaction_values(X_train_scaled)
shap.summary_plot(shap_interaction_values, X_train_scaled, show=False, plot_size=[3.44, 2.56],feature_names=dataset.columns,
                  color_bar=False, cmap='RdBu', alpha=0.9)
# plot_type='layered_violin'  使用layered_violin没法重新显示colorbar
# show=False关闭SHAP函数的绘图参数, plot_size单位为英寸
'''

# dependence_plot
'''
explainer2 = shap.TreeExplainer(regressor)
shap_values = explainer2.shap_interaction_values(X_train_scaled)
# print(shap_values)
shap_values_save = pd.DataFrame(shap_values.reshape(2443*17,17))
shap_values_save.to_csv('shap_value_NNI.csv')  # 另存为csv文件

shap.dependence_plot(('AGDD','RESAVI'),shap_values, X_train_scaled, feature_names=dataset.columns, show=False,
                     cmap='RdBu',
                     dot_size=10, alpha=0.8)  # Feature从0开始按顺序排列
'''

'''
fig = plt.gcf()  # gcf means "get current figure"
ax = plt.gca()  # gca means "get current axes"
#fig.set_figwidth(3.34)
#fig.set_figheight(2.56)

cb = plt.colorbar()  # 重新显示colorbar并实现控制
cb.set_label('Feature Value', fontsize=8)  # colorbar标记
cb.ax.tick_params(labelsize=8, width=0.2)  # 设置colorbar刻度的字体大小
cb.outline.set_linewidth(0)  # colorbar外框

#ax.set_xlim(-6000,6000)  # 图片尺寸太小会导致坐标显示异常，可以自定义设置
#ax.set_xlabel('SHAP Value', fontsize=8)  # 横坐标轴标题和字体大小
# ax.set_ylabel('SHAP Value for AGDD', fontsize=8)
#ax.tick_params(labelsize=8)  # 标签大小
# ax.tick_params(axis='y', pad=-10)  # 默认的 yticklabel 偏离太远
#ax.spines['right'].set_visible(True)
#ax.spines['left'].set_visible(True)
#ax.spines['top'].set_visible(True)  # 把左右上的边框显示出来（默认是不显示）
#plt.title('AGB-SHAP', fontsize=8)  # 标题和字体大小
plt.tight_layout()
# plt.subplots_adjust(right=0.96)  # PNC的图需要微调
plt.savefig('AGB-SHAP-Interaction.jpeg', dpi=500)
plt.show()
'''
# 特征重要性-基于feature importance

