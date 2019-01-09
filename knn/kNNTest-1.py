#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# In[3]:


import pandas as pd
import numpy as np

# In[4]:


train_main = pd.read_csv('../input/train.csv')
test_main = pd.read_csv('../input/test.csv')

# In[5]:


from sklearn.model_selection import train_test_split

# In[6]:


import matplotlib.pyplot as plt

first_digit = train_main.iloc[1].drop('label').values.reshape(28, 28)
plt.imshow(first_digit)
# plt.show()

# In[7]:


train, test = train_test_split(train_main, test_size=0.3, random_state=100)

# In[8]:


train_x1 = train.drop('label', axis=1)
train_y1 = train['label']

test_x1 = test.drop('label', axis=1)
test_y1 = test['label']

# In[9]:


# DECISION TREE, RANDOM FOREST, KNN, ADA BOOS TOGETHER

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

df = []
# model_1 = DecisionTreeClassifier(random_state=100, max_depth=3)
# model_2 = RandomForestClassifier(random_state=100, n_estimators=300)
# model_3 = KNeighborsClassifier(n_neighbors=5)
# model_4 = AdaBoostClassifier(random_state=100, n_estimators=800)
# a = (model_1, model_2, model_3, model_4)
# for i in a:
#     i.fit(train_x1, train_y1)
#     pred_test = i.predict(test_x1)
#     a = accuracy_score(test_y1, pred_test)
#     df.append(a)
#
# print('Decision Tree :', df[0],
#       'Random Forest :', df[1],
#       'KNN :', df[2],
#       'Ada Boost :', df[3])
#
# # In[10]:
#
#
# accuracy = df[2]
# print('accuracy by knn :', df[2])

# In[11]:

model_3 = KNeighborsClassifier(n_neighbors=5)
model_3.fit(train_x1, train_y1)
pred_test = model_3.predict(test_x1)
a = accuracy_score(test_y1, pred_test)
df.append(a)
print('KNN :', df[0])

accuracy = df[0]
print('accuracy by knn :', df[0])

test_pred = model_3.predict(test_main)
df_test_pred = pd.DataFrame(test_pred, columns=['Predicted'])
df_test_pred['ImageId'] = test_main.index + 1

# In[12]:


df_test_pred[['ImageId', 'Predicted']].to_csv('submission.csv', index=False)

# In[13]:


pd.read_csv('submission.csv').head()

# In[ ]:
