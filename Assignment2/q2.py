#!/usr/bin/env python
# coding: utf-8

# In[136]:


#importing required librarires
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector


# In[137]:


#data reading
df=pd.read_csv("iris.data",names=['sepal length','sepal width','petal length','petal width','class'])
df


# In[138]:


#Standard Normalization
def maxmin(data):
    result=[]
    for i in range(data.shape[1]-1):
        fvalues=[]
        for j in range(data.shape[0]):
            fvalues.append(data.loc[j,data.columns[i]])
        min_val=min(fvalues)
        max_val=max(fvalues)
        result.append([min_val,max_val])
    return(result)
def standard_normalization(data):
    mm=maxmin(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]-1):
            data.loc[i,data.columns[j]]=(data.loc[i,data.columns[j]]-mm[j][0])/(mm[j][1]-mm[j][0])
    return(data)
        


# In[139]:


df=standard_normalization(df)
df


# In[140]:


#Sampling
def sampling(dataset,split=0.80):
    X = np.array(df.drop(['class'],axis=1))
    y = np.array(df["class"])
    n_train = math.floor(split * dataset.shape[0])
    n_test = math.ceil((1-split) * dataset.shape[0])
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    return X_train,X_test,y_train,y_test


# In[141]:


X_train, X_test, y_train, y_test =sampling(df)
#SVM
#linear
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print("Accuracy score of linear kernel:",accuracy_score(y_test,y_pred))
#quadratic
svclassifier = SVC(kernel='poly',degree=2)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print("Accuracy score of quadratic kernel:",accuracy_score(y_test,y_pred))
#
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print("Accuracy score of Radial Basis function kernel:",accuracy_score(y_test,y_pred))


# In[142]:


#MLP Classifier
X=df.drop(['class'],axis=1)
y=df['class']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
#16 hidden nodes
model1 = MLPClassifier(hidden_layer_sizes=(16),solver='sgd',learning_rate_init=0.001,batch_size=32,max_iter=200,random_state=0).fit(X_train, y_train)
y_pred_1 = model1.predict(X_test)
m1=accuracy_score(y_test,y_pred_1)
print("Accuracy of MLPClassifier with 1 hidden layer with 16 nodes :", m1)
#256-16 hidden nodes
model2 = MLPClassifier(hidden_layer_sizes=(256,16),solver='sgd',learning_rate_init=0.001,batch_size=32,max_iter=200,random_state=0).fit(X_train, y_train)
y_pred_2 = model2.predict(X_test)
m2=accuracy_score(y_test,y_pred_2)
print("Accuracy of MLPClassifier with 2 hidden layers with 256 and 16 nodes :", m2)


# In[143]:


#learning rate vs accuracy
learning_rates=[0.1, 0.01, 0.001, 0.0001, 0.00001]
accuracy=[]
hidden_layer=()
m=max(m1,m2)
flag=0
if(m==m1):
    hidden_layer=(16)
    flag=1
else:
    hidden_layer=(256,16)
    flag=2
for i in learning_rates:
    clf=MLPClassifier(hidden_layer_sizes=hidden_layer,solver='sgd',learning_rate_init=i,batch_size=32,max_iter=200,random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy.append(accuracy_score(y_test,y_pred))

plt.plot(learning_rates,accuracy)
plt.title('Learning rate Vs Accuracy')
plt.xlabel('Learning rate')
plt.ylabel('Accuracy')
plt.show()
    
                         
    


# In[144]:


#backward elimation method
clf=MLPClassifier(hidden_layer_sizes=hidden_layer,solver='sgd',learning_rate_init=0.001,batch_size=32,max_iter=1000,random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)
back_feature_select=SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),k_features=4,forward=False,scoring="accuracy").fit(X_train,y_train)
back_feature_select.k_feature_names_


# In[145]:


#ensemble learning
from sklearn.ensemble import VotingClassifier
svm_quad=SVC(kernel='poly',degree=2)
svm_rbf=SVC(kernel='rbf')
if(flag==1):
    best_of_part3=MLPClassifier(hidden_layer_sizes=(16),solver='sgd',learning_rate_init=0.001,batch_size=32,max_iter=100,random_state=0)
else:
    best_of_part3=MLPClassifier(hidden_layer_sizes=(256,16),solver='sgd',learning_rate_init=0.001,batch_size=32,max_iter=100,random_state=0)

final_model = VotingClassifier(estimators=[('svc', svm_rbf),('MLP', best_of_part3)], voting='hard')

svm_quad.fit(X_train, y_train)
pred_svm_quad = svm_quad.predict(X_test)
print("Accuracy of SVM using quadratic polynomial kernel:",accuracy_score(y_test,pred_svm_quad))

svm_rbf.fit(X_train, y_train)
pred_svm_rbf = svm_rbf.predict(X_test)
print("Accuracy of SVM using Radial Basis function kernel:",accuracy_score(y_test,pred_svm_rbf))

best_of_part3.fit(X_train, y_train)
pred_best_of_part3 = best_of_part3.predict(X_test)
print("Accuracy of best model of part3:",accuracy_score(y_test,pred_best_of_part3))


final_model.fit(X_train, y_train)
pred_final = final_model.predict(X_test)
print("Accuracy of max voting technique using the above 3 models:",accuracy_score(y_test, pred_final))


# In[ ]:





# In[ ]:




