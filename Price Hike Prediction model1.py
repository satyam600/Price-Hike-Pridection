#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics


# In[13]:


dataset=pd.read_csv("Price1.csv")
print(dataset.head())


# In[14]:


dataset.info()


# In[15]:


zero_not_accepted=['Branch','Unit price','Quantity','cogs',"gross income"]
for column in zero_not_accepted:
    dataset[column]=dataset[column].replace(0, np.NaN)
    mean=int(dataset[column].mean(skipna=True))
    dataset[column]=dataset[column].replace(np.NaN, mean)


# In[16]:


print(dataset['cogs'])


# In[17]:


X=dataset.iloc[:, 0:5]
y=dataset.iloc[:, 5]
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0, test_size=0.2)


# In[18]:


sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


# In[27]:


classifier=KNeighborsClassifier(n_neighbors=15, p=2, metric='euclidean')


# In[28]:


classifier.fit(X_train, y_train)


# In[29]:


y_pred=classifier.predict(X_test)
y_pred


# In[30]:


cm=confusion_matrix(y_test, y_pred)
plt.show()
print(cm)


# In[31]:


print(f1_score(y_test, y_pred))


# In[32]:


print(accuracy_score(y_test, y_pred))


# In[ ]:




