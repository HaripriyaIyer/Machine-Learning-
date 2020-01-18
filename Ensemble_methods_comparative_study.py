#!/usr/bin/env python
# coding: utf-8

# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score


# In[3]:


# Read the training dataset

train_df = pd.read_csv('train.csv')
# Find the correlation between the features and the target
train_df.corr()


# In[4]:


# Create a heatmap of the correlations for a better analysis
ax = sns.heatmap(
    train_df.corr(), 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[5]:


target = train_df['target']


# In[7]:


#Drop redundant features
train_df_X = train_df.drop(columns=['ID', 'age','target'])


# In[8]:


#Get only the useful features from the train set and convert cateogrical values into ordinal values by one hot encoding
c = ['job', 'marital', 'education', 'connect',
       'landline', 'smart', 'last_month', 'poutcome']
df_processed = pd.get_dummies(train_df_X, prefix_sep="__",
                              columns=c)
df_processed.columns


# In[9]:


df_processed.head()


# In[13]:


#Split data into training and test sets    
X_train, X_test, y_train, y_test = train_test_split(df_processed, target, test_size=0.33, random_state=42)


# In[16]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

clf_rf = RandomForestClassifier(max_depth=4, random_state=0)
clf_rf.fit(X_train, y_train)

clf_b = BaggingClassifier(n_estimators=20, random_state=0)
clf_b.fit(X_train, y_train)

clf_gb = GradientBoostingClassifier(n_estimators= 10, max_leaf_nodes= 4)
clf_gb.fit(X_train, y_train)


# In[17]:


y_predict = clf.predict(X_test)
y_predict_rf = clf_rf.predict(X_test)
y_predict_b = clf_b.predict(X_test)
y_predict_gb = clf_gb.predict(X_test)



print(accuracy_score(y_test, y_predict))
print(accuracy_score(y_test, y_predict_rf))
print(accuracy_score(y_test, y_predict_b))
print(accuracy_score(y_test, y_predict_gb))


# In[18]:


from sklearn.linear_model import LogisticRegression


# In[20]:


clf_lr = LogisticRegression(penalty = 'l2',random_state=0).fit(X_train, y_train)
y_predict_lr = clf_lr.predict(X_test)


# In[21]:


accuracy_score(y_test, y_predict_lr)


# In[ ]:




