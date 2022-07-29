#  -*- coding = utf-8 -*- 
#  @time2022/4/1511:21
#  Author:Ruanguojie

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor  # 多输出回归
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
# 画图模块
import matplotlib.pyplot as plt
import seaborn as sns
import palettable


# 读取数据
dataset = pd.read_csv('N/mtl.csv')
# 打乱数据集：
dataset = dataset.sample(frac=1.0, random_state=42)
data = dataset.reset_index(drop=True)
# Dataset Analysis and Visualization
# display first 5 rows of dataset to see what we are looking at
print(dataset.head())
# show the distributions of data
print(dataset.describe())

#  直观地看见标记是什么
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

# 重新将标记叠加
y_train = np.hstack((y1_train, y2_train, y3_train, y4_train))
y_test = np.hstack((y1_test, y2_test, y3_test, y4_test))
print(y_train.shape)
print(y_test.shape)

# StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
mean_y1_test = np.mean(y1_test)
mean_y2_test = np.mean(y2_test)
mean_y3_test = np.mean(y3_test)
mean_y4_test = np.mean(y4_test)

MTLRF = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
)
MTLRF = MTLRF.fit(X_train_scaled, y_train)

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
joblib.dump(MTLRF, 'save/MTLRF.pkl')  # 更改文件名
# restore
regressor = joblib.load('save/MTLRF.pkl')

y_pred_RF = regressor.predict(X_test_scaled)
RF_train_score = regressor.score(X_train_scaled, y_train)
RF_test_score = r2_score(y_test, y_pred_RF)
RF_test_score_AGB = r2_score(y_test[:, 0], y_pred_RF[:, 0])
RF_test_score_PNC = r2_score(y_test[:, 1], y_pred_RF[:, 1])
RF_test_score_PNU = r2_score(y_test[:, 2], y_pred_RF[:, 2])
RF_test_score_NNI = r2_score(y_test[:, 3], y_pred_RF[:, 3])
RF_test_MSE_AGB = MSE(y_test[:, 0], y_pred_RF[:, 0])
RF_test_MSE_PNC = MSE(y_test[:, 1], y_pred_RF[:, 1])
RF_test_MSE_PNU = MSE(y_test[:, 2], y_pred_RF[:, 2])
RF_test_MSE_NNI = MSE(y_test[:, 3], y_pred_RF[:, 3])
RMSE_test_AGB = RF_test_MSE_AGB ** 0.5
RMSE_test_PNC = RF_test_MSE_PNC ** 0.5
RMSE_test_PNU = RF_test_MSE_PNU ** 0.5
RMSE_test_NNI = RF_test_MSE_NNI ** 0.5
rRMSE_AGB = RMSE_test_AGB/mean_y1_test
rRMSE_PNC = RMSE_test_PNC/mean_y2_test
rRMSE_PNU = RMSE_test_PNU/mean_y3_test
rRMSE_NNI = RMSE_test_NNI/mean_y4_test


print("r^2 of RF regression on training data: %.4f" % RF_train_score)
print("r^2 of RF regression on test data: %.4f" % RF_test_score)  # 即4个任务的R2平均值
print("r^2 of RF regression on test data(AGB): %.4f" % RF_test_score_AGB)
print("r^2 of RF regression on test data(PNC): %.4f" % RF_test_score_PNC)
print("r^2 of RF regression on test data(PNU): %.4f" % RF_test_score_PNU)
print("r^2 of RF regression on test data(NNI): %.4f" % RF_test_score_NNI)
print("RMSE of RF regression on test data(AGB): %.4f" % RMSE_test_AGB)
print("RMSE of RF regression on test data(PNC): %.4f" % RMSE_test_PNC)
print("RMSE of RF regression on test data(PNU): %.4f" % RMSE_test_PNU)
print("RMSE of RF regression on test data(NNI): %.4f" % RMSE_test_NNI)
print("rRMSE of RF regression on test data(AGB): %.4f" % rRMSE_AGB)
print("rRMSE of RF regression on test data(PNC): %.4f" % rRMSE_PNC)
print("rRMSE of RF regression on test data(PNU): %.4f" % rRMSE_PNU)
print("rRMSE of RF regression on test data(NNI): %.4f" % rRMSE_NNI)

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
plt.xlabel('NNI_Measured', fontsize=10)
plt.ylabel('NNI_Predicted', fontsize=10)  # 横纵坐标轴标题
plt.title('NNI', fontsize=10)  # 图表标题
plt.tight_layout()
plt.subplots_adjust(left=0.15)
plt.savefig('NNI1.jpeg')
plt.show()
'''
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
cb.ax.tick_params(labelsize=8, width=0.5)  # 设置图例刻度的字体大小
cb.outline.set_linewidth(0.5)  # 图例外框
# cb.set_ticks(np.linspace(1, 10, 10))
plt.xlabel('NNI_Measured', fontsize=10)  # 横坐标标题
plt.ylabel('NNI_Predicted', fontsize=10)  # 纵坐标轴标题
plt.title('NNI', fontsize=10)  # 图表标题
'''

# 添加文本框
'''
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

'''
plt.tight_layout()
plt.subplots_adjust(left=0.15)
plt.savefig('NNI2.jpeg')
plt.show()
'''