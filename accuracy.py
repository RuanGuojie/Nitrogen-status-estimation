# -*- coding = utf-8 -*-
# @Time： 2022/6/24 19:26
# @Author： Ruanguojie

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, cohen_kappa_score, precision_score, f1_score
import pandas as pd
import seaborn as sns
data = pd.read_csv('KAPPA_DNN.csv')
y_test = data['Diagnosis']
y_pred = data['Prediction']


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ConfusionMatrixDisplay.from_predictions(y_test, y_pred,labels=['deficient','optimal','surplus'],cmap='Blues')
fig = plt.gcf()
ax = plt.gca()

fig.set_figwidth(4)
fig.set_figheight(3)
ax.tick_params(labelsize=12)
ax.set_xlabel('Predicted label',size=12)
ax.set_ylabel('True label',size=12)
plt.tight_layout()
plt.savefig('DNN.jpeg',dpi=500)
plt.show()



'''
cf_matrix = confusion_matrix(y_true=y_test,y_pred=y_pred)
ax = sns.heatmap(cf_matrix, cmap='Blues', annot=True,linewidths=1)
ax.yaxis.set_ticklabels(['deficient','optimal','surplus'])
ax.xaxis.set_ticklabels(['deficient','optimal','surplus'])
plt.show()
'''


accuracy = accuracy_score(y_true=y_test,y_pred=y_pred)
print(accuracy)
kappa = cohen_kappa_score(y_test,y_pred)
print(kappa)
precision = precision_score(y_true=y_test,y_pred=y_pred,average='weighted')
print(precision)
f1_score = f1_score(y_true=y_test,y_pred=y_pred,average='weighted')
print(f1_score)