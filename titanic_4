# 출처: https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
sns.set(style='white', context='notebook', palette='deep')

PATH = "/home/workspace/dacon/titanic/data/"


# In[3]:


# Load data
##### Load train and Test set

train = pd.read_csv(PATH + "train.csv")
test = pd.read_csv(PATH + "test.csv")
IDtest = test["PassengerId"]


# In[4]:


train.info()


# In[5]:


train[["Age","SibSp","Parch","Fare"]].head()


# In[6]:


# Outlier detection 

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
                
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])


# In[7]:


train.loc[Outliers_to_drop]


# In[8]:


train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


# In[ ]:





# In[9]:


train_len = len(train)
dataset = pd.concat(objs = [train, test], axis = 0).reset_index(drop=True)


# In[ ]:





# In[10]:


dataset = dataset.fillna(np.nan)
dataset.isna().sum()


# In[ ]:





# In[11]:


train.info()


# In[12]:


train.isnull().sum()


# In[13]:


train.head()


# In[14]:


train.dtypes


# In[15]:


train.describe()


# In[ ]:





# ## Numerical values

# In[16]:


g = sns.heatmap(train[["Survived", "SibSp", "Parch", "Age", "Fare"]].corr(), annot=True, fmt = ".2f", cmap = "coolwarm")


# In[ ]:





# In[17]:


#SibSp


# In[18]:


g = sns.factorplot(x = "SibSp", y="Survived", data = train, kind = 'bar', size = 6, palette = "muted")
g.despine(left = True)
g = g.set_ylabels("survival probability")


# In[ ]:





# In[19]:


#Parch


# In[20]:


g = sns.factorplot(x= "Parch", y= "Survived", data=train, kind = "bar", size = 6, palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[ ]:





# In[21]:


g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")


# In[22]:


g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color = "Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax = g, color = "blue", shade = True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived", "Survived"])


# In[ ]:





# In[23]:


# Fare


# In[24]:


dataset["Fare"].isnull().sum()


# In[25]:


dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())


# In[ ]:





# In[26]:


g = sns.distplot(dataset["Fare"], color = "m", label = "Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")


# In[27]:


dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)


# In[ ]:





# In[28]:


g = sns.distplot(dataset['Fare'], color = "b", label = "Skewness: %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc = "best")


# In[ ]:





# In[ ]:





# ## Categorical Value

# In[29]:


g = sns.barplot(x="Sex", y="Survived", data=train)
g = g.set_ylabel("Survival Probability")


# In[30]:


train[["Sex", "Survived"]].groupby('Sex').mean()


# In[ ]:





# In[31]:


# Pclass


# In[32]:


g = sns.factorplot(x = "Pclass",  y="Survived", data= train, kind="bar", size = 6, palette = "muted")
g.despine(left = True)
g = g.set_ylabels("survival probability")


# In[ ]:





# In[33]:


g = sns.factorplot(x = "Pclass", y="Survived",  hue="Sex", data=train, size = 6, kind = "bar", palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[ ]:





# In[34]:


# Embarked


# In[35]:


dataset["Embarked"].isnull().sum()


# In[36]:


dataset["Embarked"] = dataset["Embarked"].fillna("S")


# In[ ]:





# In[37]:


g = sns.factorplot(x = "Embarked", y="Survived", data=train, size=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Survival probability")


# In[ ]:





# In[38]:


g = sns.factorplot("Pclass", col="Embarked", data=train, size = 6, kind = "count", palette ="muted")
g.despine(left=True)
g = g.set_ylabels("Count")


# In[ ]:





# In[ ]:





# # 4.Filling missing Values

# In[39]:


g = sns.factorplot(y = "Age", x = "Sex", data = dataset, kind = "box")
g = sns.factorplot(y = "Age", x = "Sex", hue = "Pclass", data= dataset, kind = "box")
g = sns.factorplot(y = "Age", x = "Parch", data= dataset, kind = "box")
g = sns.factorplot(y = "Age", x = "SibSp", data= dataset, kind = "box")


# In[ ]:





# In[40]:


dataset["Sex"] = dataset["Sex"].map({"male":0, "female":1})


# In[ ]:





# In[41]:


g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)


# In[ ]:





# In[42]:


index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in index_NaN_age:
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred):
        dataset["Age"].iloc[i] = age_pred
    else : 
        dataset["Age"].iloc[i] = age_med


# In[43]:


g = sns.factorplot(x = "Survived", y="Age", data=train, kind = "box")
g = sns.factorplot(x = "Survived", y="Age", data=train, kind = "violin")


# In[ ]:





# In[ ]:





# # 5.Feature engineering

# In[44]:


dataset['Name'].head()


# In[45]:


dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]


# In[46]:


dataset["Title"] = pd.Series(dataset_title)
dataset["Title"].head()


# In[47]:


g = sns.countplot(x = "Title", data = dataset)
g = plt.setp(g.get_xticklabels(), rotation=45)


# In[48]:


dataset['Title'].value_counts()


# In[49]:


dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)


# In[50]:


dataset['Title'].value_counts()


# In[51]:


g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])


# In[52]:


g = sns.factorplot(x = "Title", y = "Survived", data = dataset, kind="bar")
g = g.set_xticklabels(["Master", "Miss-Mrs", "Mr", "Rare"])
g = g.set_ylabels("survival probability")


# In[53]:


dataset.drop(labels = ["Name"], axis = 1, inplace = True)


# In[ ]:





# In[54]:


# Family size


# In[55]:


dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1


# In[56]:


g = sns.factorplot(x = "Fsize", y="Survived", data = dataset)
g = g.set_ylabels("Survival Probability")


# In[ ]:





# In[57]:


dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if s == 2 else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)


# In[58]:


g = sns.factorplot(x = "Single", y="Survived", data=dataset, kind = "bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x = "SmallF", y="Survived", data=dataset, kind = "bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x = "MedF", y="Survived", data=dataset, kind = "bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x = "LargeF", y="Survived", data=dataset, kind = "bar")
g = g.set_ylabels("Survival Probability")


# In[59]:


dataset = pd.get_dummies(dataset, columns = ["Title"])
dataset = pd.get_dummies(dataset, columns = ['Embarked'], prefix = "Em") 


# In[60]:


dataset.head()


# In[ ]:





# In[61]:


dataset["Cabin"].head()


# In[62]:


dataset["Cabin"].describe()


# In[63]:


dataset["Cabin"].isnull().sum()


# In[64]:


dataset["Cabin"][dataset["Cabin"].notnull()].head()


# In[65]:


dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin']])


# In[66]:


g = sns.countplot(dataset["Cabin"],order=['A','B','C','D','E','F','G','T','X'])


# In[67]:


g = sns.factorplot(x = "Cabin", y = "Survived", data = dataset, kind="bar", order=['A','B','C','D','E','F','G','T','X'])
g = g.set_ylabels("Survival Probability")


# In[68]:


dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")


# In[ ]:





# In[69]:


dataset["Ticket"].head()


# In[70]:


Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit():
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0])
    else:
        Ticket.append("X")

dataset["Ticket"] = Ticket
dataset["Ticket"].head()


# In[71]:


dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix ="T")


# In[72]:


dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"], prefix = "Pc")


# In[73]:


dataset.drop(labels = ["PassengerId"], axis =1, inplace = True)


# In[74]:


dataset.head()


# In[ ]:





# # 6. Modeling

# In[ ]:





# In[75]:


train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels = ["Survived"], axis =1, inplace=True)


# In[76]:


train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]
X_train = train.drop(labels=["Survived"], axis =1)


# In[77]:


len(X_train)


# In[ ]:





# In[78]:


kfold = StratifiedKFold(n_splits = 10)


# In[79]:


# Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# In[80]:


# def save_ranking(*args, **kwargs):
#     print(args)
#     print(kwargs)
# save_ranking('ming', 'alice', 'tom', fourth='wilson', fifth='roy')


# In[81]:


DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
                 "base_estimator__splitter" : ["best", "random"],
                 "algorithm" : ["SAMME", "SAMME.R"],
                 "n_estimators" : [1, 2],
                 "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}

gsadaDTC = GridSearchCV(adaDTC, param_grid = ada_param_grid, cv = kfold, scoring = "accuracy", n_jobs = 4, verbose = 1)

gsadaDTC.fit(X_train, Y_train)

ada_best = gsadaDTC.best_estimator_


# In[82]:


gsadaDTC.best_score_


# In[83]:


ada_best


# In[ ]:





# In[84]:


ExtC = ExtraTreesClassifier()

ex_param_grid = {"max_depth" : [None],
                "max_features" : [1, 3, 10],
                "min_samples_split" : [2, 3, 10],
                "min_samples_leaf" : [1, 3, 10],
                "bootstrap" : [False],
                "n_estimators" : [100, 300],
                "criterion" : ["gini"]}

gsExtC = GridSearchCV(ExtC, param_grid = ex_param_grid, cv=kfold, scoring = "accuracy", n_jobs = 4, verbose = 1)

gsExtC.fit(X_train, Y_train)

ExtC_best = gsExtC.best_estimator_

gsExtC.best_score_


# In[ ]:





# In[85]:


RFC = RandomForestClassifier()

rf_param_grid = {"max_depth" : [None],
                "max_features" : [1, 3, 10], 
                "min_samples_split" : [2, 3, 10],
                "min_samples_leaf" : [1, 3, 10],
                "bootstrap" : [False],
                "n_estimators" : [100, 300],
                "criterion" : ["gini"]}

gsRFC = GridSearchCV(RFC, param_grid = rf_param_grid, cv = kfold, scoring = "accuracy", n_jobs = 4, verbose = 1)

gsRFC.fit(X_train, Y_train)

RFC_best = gsRFC.best_estimator_

gsRFC.best_score_


# In[ ]:





# In[86]:


GBC = GradientBoostingClassifier()

gb_param_grid = {"loss" : ["deviance"],
                "n_estimators" : [100, 200, 300],
                "learning_rate" : [0.1, 0.05, 0.01],
                "max_depth" : [4, 8],
                "min_samples_leaf" : [100, 150],
                "max_features" : [0.3, 0.1]}

gsGBC = GridSearchCV(GBC, param_grid = gb_param_grid, cv = kfold, scoring ="accuracy", n_jobs=4, verbose=1 )

gsGBC.fit(X_train, Y_train)

GBC_best = gsGBC.best_estimator_

gsGBC.best_score_


# In[ ]:





# In[87]:


SVMC = SVC(probability=True)

svc_param_grid = {'kernel' : ['rbf'],
                 'gamma' : [0.001, 0.01, 0.1, 1],
                 'C' : [1, 10, 50, 100, 200, 300, 1000]}

gsSVMC = GridSearchCV(SVMC, param_grid = svc_param_grid, cv = kfold, scoring="accuracy", n_jobs =4, verbose=1)

gsSVMC.fit(X_train, Y_train)

SVMC_best = gsSVMC.best_estimator_

gsSVMC.best_score_


# In[90]:


def plot_learning_curve(estimator, title
                        , X, y, ylim = None, cv = None, 
                       n_jobs = -1, train_sizes = np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv = cv, n_jobs = n_jobs, train_sizes = train_sizes)

    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)

    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.std(test_scores, axis= 1)

    plt.grid()

    plt.fill_between(train_sizes, 
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha = 0.1, 
                     color = "r")

    plt.fill_between(train_sizes,
                    test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std,
                    alpha = 0.1, 
                    color = "g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean,  'o-', color="g", label="Cross-validation score")

    plt.legend(loc = "best")

    return plt


# In[91]:


g = plot_learning_curve(gsRFC.best_estimator_,"RF learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",X_train,Y_train,cv=kfold)


# In[ ]:





# In[ ]:





# In[111]:


nrows = ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex = "all", figsize = (15, 15))

names_classifiers = [("AdaBoosting", ada_best),
                     ("ExtraTrees", ExtC_best),
                     ("RandomForest", RFC_best),
                     ("GradientBoosting", GBC_best)]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        
        g = sns.barplot(y = X_train.columns[indices][:40], 
                        x = classifier.feature_importances_[indices][:40],
                        orient = 'h', ax = axes[row][col])
        g.set_xlabel("Relative importance", fontsize = 12)
        g.set_ylabel("Features", fontsize =12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        nclassifier += 1
        


# In[112]:


test_Survived_RFC = pd.Series(RFC_best.predict(test), name = "RFC")
test_Survived_ExtC = pd.Series(ExtC_best.predict(test), name="ExtC")
test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name="SVC")
test_Survived_AdaC = pd.Series(ada_best.predict(test), name="Ada")
test_Survived_GBC = pd.Series(GBC_best.predict(test), name="GBC")


# In[116]:


ensemble_results = pd.concat([test_Survived_RFC,test_Survived_ExtC,test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC],axis=1)
# ensemble_results


# In[117]:


g= sns.heatmap(ensemble_results.corr(),annot=True)


# In[ ]:





# In[ ]:


# Ensemble


# In[119]:


votingC = VotingClassifier(estimators = [('rfc', RFC_best), ('extc', ExtC_best), ('svc', SVMC_best), 
                                         ('adac',ada_best),('gbc',GBC_best)], 
                           voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, Y_train)


# In[120]:


votingC


# In[ ]:


# Prediction


# In[127]:


test_Survived = pd.Series(votingC.predict(test), name="Survived")
results = pd.concat([IDtest, test_Survived], axis = 1)
results.to_csv("ensemble_python_vlting.csv", index=False)


# In[128]:


results


# In[ ]:




