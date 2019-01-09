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

original_digit = train_main.iloc[1].values
print(original_digit)

zero_digit = train_main.iloc[0].drop('label').values.reshape(28, 28)
plt.imshow(zero_digit)
# plt.show()

first_digit = train_main.iloc[130].drop('label').values.reshape(28, 28)
plt.imshow(first_digit)
# plt.show()
