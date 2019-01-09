# 本例子下面使用了多种算法（决策树、随机森林、K近邻..），计算较慢
# KNN的可以跑 kNNTest-1.py 只使用了knn
# 参考 https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

import numpy as np
import pandas as pd
import os

print(os.listdir("../input"))
train_main = pd.read_csv('../input/train.csv')
test_main = pd.read_csv('../input/test.csv')

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

first_digit = train_main.iloc[1].drop('label').values.reshape(28, 28)
plt.imshow(first_digit)

train, test = train_test_split(train_main, test_size=0.3, random_state=100)
train_x1 = train.drop('label', axis=1)
train_y1 = train['label']

test_x1 = test.drop('label', axis=1)
test_y1 = test['label']

# DECISION TREE, RANDOM FOREST, KNN, ADA BOOS TOGETHER

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

df = []
model_1 = DecisionTreeClassifier(random_state=100, max_depth=3)
model_2 = RandomForestClassifier(random_state=100, n_estimators=300)
model_3 = KNeighborsClassifier(n_neighbors=5)
model_4 = AdaBoostClassifier(random_state=100, n_estimators=800)
a = (model_1, model_2, model_3, model_4)
for i in a:
    i.fit(train_x1, train_y1)   # 使用X作为训练数据并使用y作为目标值来拟合模型
    pred_test = i.predict(test_x1)  # 预测所提供数据的类标签
    a = accuracy_score(test_y1, pred_test)
    df.append(a)

print('Decision Tree :', df[0],
      'Random Forest :', df[1],
      'KNN :', df[2],
      'Ada Boost :', df[3])

test_pred = model_3.predict(test_main)
df_test_pred = pd.DataFrame(test_pred, columns=['Predicted'])
df_test_pred['ImageId'] = test_main.index + 1
df_test_pred[['ImageId', 'Predicted']].to_csv('submission.csv', index=False)
pd.read_csv('submission.csv').head()
