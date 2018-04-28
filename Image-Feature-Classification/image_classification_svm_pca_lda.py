
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.metrics import classification_report as cls_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

from sklearn.externals import joblib

np.random.seed(42)


# In[2]:


train = pd.read_csv('data/train.dat', ' ', header=None)
train_label = pd.read_csv('data/train.labels', header=None, names=['labels'])

test = pd.read_csv('data/test.dat', ' ', header=None)

X_train = train.values
y_train = train_label.values.ravel()

X_test = test.values


# In[3]:


feats_idx = {
    'HoG':  [0, 512],    # 512
    'Hist': [512, 768],  # 256
    'LBP':  [768, 832],  # 64
    'RGB':  [832, 880],  # 48 
    'DF':   [880, 887],  # 7
}


# In[4]:


X_train = X_train[:, 832:880]
X_test = X_test[:, 832:880]

y_cls = np.unique(y_train)


# In[5]:


pca = PCA(n_components=0.95)
lda = LDA()


# In[6]:


X_train_pc = pca.fit_transform(X_train)
X_train_lda = lda.fit_transform(X_train, y_train)


# In[7]:


X_test_pc = pca.transform(X_test)
X_test_lda = lda.transform(X_test)


# In[8]:


# x_train = X_train
# x_test = X_test

# x_train = X_train_pc
# x_test = X_test_pc

# x_train = X_train_lda
# x_test = X_test_lda

x_train = np.hstack((X_train_pc, X_train_lda))
x_test = np.hstack((X_test_pc, X_test_lda))


# In[9]:


svm = SVC(C=5, gamma=0.5, cache_size=10000, class_weight='balanced')


# In[10]:


svm = svm.fit(x_train, y_train)


# In[11]:


y_pred = svm.predict(x_train)


# In[12]:


print(cls_report(y_train, y_pred))
print(confusion_matrix(y_train, y_pred))


# In[13]:


y_test_pred = svm.predict(x_test)


# In[14]:


result = pd.DataFrame(data=y_test_pred)
fn = 'svm_rbf_c_5_gamma_0p5_pca_0p95_lda.dat'
result.to_csv(fn, header=None, index=False)

