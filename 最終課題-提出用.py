#!/usr/bin/env python
# coding: utf-8

# In[1]:


#線形識別
#SVM
#multilayer perceptron
#comper with their Accuracy


# In[2]:


import numpy as np
from sklearn.neural_network import MLPClassifier

from sklearn import datasets


# In[8]:


cancer = datasets.load_breast_cancer()
#データの読み込み


# In[9]:


X = cancer.data
print(X.shape);
d = cancer.target
print(d.shape);
#指標の入力　xは変数の集合　
#569人にで30個の指標　横は569、縦は30　0が陰性、1が陽性


# In[10]:


print( (X.T@X).shape )
print( np.linalg.matrix_rank(X.T@X) )
##行列の行と列を入れ替える
#RANK関数でデータが廃棄指標のないことを確認する　線型独立の確保


# In[12]:


import pandas as pd
print(d)


# In[14]:


y = pd.cut(d, bins = 2)
print(y)
#二項分類　データが連続値であれば分類を行う


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm


# In[17]:


#逆行列
IXX = np.linalg.inv(X.T@X)
#最小二乗法で係数を決定
b = np.linalg.inv(X.T@X)@X.T@d
#求めた係数を使って実際の出力値を計算
y = X@b

print(y)


# In[18]:


y = list([0 if yy<=0.5 else 1 for yy in y])
print(y)
#yを0、1に戻す


# In[19]:


from sklearn.metrics import accuracy_score
#dとyによる評価結果
 
print('Class labels:', np.unique(d))
print('Misclassified samples: %d' % (d != y).sum()) #dノットイコールyの数
print('Accuracy: %.2f' % accuracy_score(d, y)) #正確さ


# In[20]:


mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=10000, tol=0.0001, random_state=1)
#mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=10000, tol=0.00001, random_state=None)
 
mlp.fit(X, d)

y = mlp.predict(X)
 
from sklearn.metrics import accuracy_score
 
print('Class labels:', np.unique(d))
print('Misclassified samples: %d' % (d != y).sum())
print('Accuracy: %.2f' % accuracy_score(d, y))

#mlpモデルを使用


# In[25]:


from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1, gamma='auto')
svm.fit(X, d)
y = svm.predict(X)
print('Class labels:', np.unique(d))
print('Misclassified samples: %d' % (d != y).sum())
print('SVM Accuracy: %.2f' % accuracy_score(d, y))

#SVMを使用


# In[24]:


accu_scores = []
data_splits = range(1,30)
for i in data_splits:
    mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=10000, tol=0.0001, random_state=None)
    x_part = X[0:len(y)*i//10,:]
    d_part = d[0:len(y)*i//10]
    mlp.fit(x_part, d_part)
    y = mlp.predict(X)
    accu_scores.append(accuracy_score(d, y))
plt.scatter(data_splits, accu_scores, c='red', s=100, label='legend')

#学習曲線を描く
#トレーニングセットの増加により、モデルの精度がどのように増加しているかを観察する


# In[ ]:


#まとめ：
#線形識別 0.96
#MLP 0.95
#SVM 0.97
#データセットは特徴的で、線形識別が0.96に達することができ、mlpより高い数値を得た


# In[ ]:




