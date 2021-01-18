#!/usr/bin/env python
# coding: utf-8

# In[37]:


# https://www.kaggle.com/sinakhorami/titanic-best-working-classifier


# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import re as re

train = pd.read_csv('/home/workspace/dacon/titanic/data/train.csv',  header = 0, dtype={'Age': np.float64})
test = pd.read_csv('/home/workspace/dacon/titanic/data/test.csv',  header = 0, dtype={'Age': np.float64})
full_data = [train, test]

print(train.info())


# # Feature Engineering

# In[2]:


## Pclass


# In[3]:


print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())


# In[4]:


## Sex


# In[5]:


print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())


# In[6]:


## SibSp and Parch = Family Size


# In[7]:


for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 #1은 본인

print(train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean())


# In[8]:


## alone


# In[9]:


for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print(train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())


# In[10]:


## Embarked


# In[11]:


for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())


# In[12]:


## Fare


# In[13]:


for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

print(train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())


# In[14]:


## Age


# In[15]:


import sys


# In[16]:


for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg-age_std, age_avg+age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)


# In[17]:


print(train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())


# In[18]:


## Name


# In[19]:


import re


# In[20]:


def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name)
    
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

print(pd.crosstab(train['Title'], train['Sex']))


# In[21]:


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt',
                                                'Col', 'Don', 'Dr', 'Major',
                                                'Rev', 'Sir', 'Jonkheer', 'Dona'],
                                               'Rare')
    
    dataset['Title'] = dataset['Title'].replace('Mile', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


# In[22]:


train.head(2)


# In[23]:


train.info()


# # Data Cleaning

# In[24]:


for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map({'female' : 0, 'male' : 1}).astype(int)
    
    title_mapping = {'Mr':1 , "Miss":2, "Mrs":3, "Master":4, "Rare":5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
    
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']

train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)

test = test.drop(drop_elements, axis = 1)

print(train.head(10))

train = train.values
test = test.values    


# # Classifier Comparison

# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


# In[26]:


classifiers = [
    KNeighborsClassifier(3),
    SVC(probability = True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]
    
log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns = log_cols)

sss = StratifiedShuffleSplit(n_splits = 10, test_size = 0.1, random_state = 0)
X = train[0::, 1::]
y = train[0::, 0]

acc_dict = {}


# In[28]:


for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions) #실제 target값, 예측값 정확도 비교
        
        if name in acc_dict:
            acc_dict[name] += acc    
        else:
            acc_dict[name] = acc
            
for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns = log_cols)
    log = log.append(log_entry)
    
plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes('muted')
sns.barplot(x='Accuracy', y='Classifier', data=log, color='b')


# In[ ]:





# In[35]:


candidate_classifier = SVC()
candidate_classifier.fit(train[0::, 1::], train[0::, 0])
result = candidate_classifier.predict(test)


# In[ ]:




