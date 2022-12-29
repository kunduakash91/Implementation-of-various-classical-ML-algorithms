#!/usr/bin/env python
# coding: utf-8

# In[341]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[342]:


df = pd.read_csv("D:\Mtech\ML\Train_B_Bayesian.csv")


# In[343]:


print(len(df))
df


# In[344]:


df.dtypes


# # Removing Outliers

# In[345]:


df['gender'] = df['gender'].map({'Male':0 ,'Female':1,'None': 2})
df['is_patient'] = df['is_patient'].map({1: 1, 2: 0})


# In[346]:


df.dtypes


# In[347]:


#Detecting the upper range
def outlier(df,column_name):
   upper_bound = (2 * df[column_name].mean()) +  (5 * df[column_name].std())
   return upper_bound


# In[348]:


df


# In[349]:


df.dtypes


# In[350]:


columns = df.columns


# In[351]:


columns


# In[352]:


for col in df.columns:
    if col!="is_patient":
        upper_bound = outlier(df,col)
        
        df.drop(df[(df[col]>upper_bound)].index,inplace = True)


# In[353]:


print(len(df))


# In[354]:


print(df)


# In[355]:


df


# # Normalize Data 

# In[356]:


for col in df.columns:
    if col!="is_patient":
        df[col] = (df[col] - df[col].min())/(df[col].max() - df[col].min())


# In[357]:


df


# # Splitting Training and Testing Data

# # Naive Bayes

# In[358]:


X = df.iloc[:,0:11].values
y=df.iloc[:,-1].values


# In[359]:


train_index = int(0.7 * len(X))
X_test, y_test =X[train_index:], y[train_index:]


# In[360]:


from random import randrange
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# In[361]:


def calculate_prior_prob(df,y):
    classes = sorted(list(df[y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[y]==i])/len(df))
    return prior


# In[362]:


def calculate_likelihood(df,col_name,value,y,label):
    df = df[df[y]==label]
    mean, std = df[col_name].mean(), df[col_name].std()
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((value-mean)**2 / (2 * std**2 )))
    return p_x_given_y


# In[363]:


def naive_bayes_gaussian(df, X, Y):
    # get feature names
    features = list(df.columns)[:-1]

    # calculate prior
    prior = calculate_prior_prob(df, Y)

    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood(df, features[i], x[i], Y, labels[j])

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred) 


# In[364]:


def accuracy_score(y_true, y_pred):
    accuracy = np.mean(y_pred == y_true)
    return accuracy


# In[365]:


dataset_length = int(len(df))
dataset_length=dataset_length//5
dataset_length


# In[366]:


fold1 = df.loc[0:dataset_length]                                            
fold2 = df.loc[dataset_length:2*dataset_length]
fold3 = df.loc[2*dataset_length:3*dataset_length]
fold4 = df.loc[3*dataset_length:4*dataset_length]
fold5 = df.loc[4*dataset_length:5*dataset_length]

train_val1 = pd.concat([fold1, fold2, fold3, fold4])
test_val1 = fold5

train_val2 = pd.concat([fold1, fold2, fold3, fold5])
test_val2 = fold4

train_val3 = pd.concat([fold1, fold2, fold4, fold5])
test_val3 = fold3

train_val4 = pd.concat([fold1, fold3, fold4, fold5])
test_val4 = fold2

train_val5 = pd.concat([fold2, fold3, fold4, fold5])
test_val5 = fold1


accuracy = []
X_test = test_val1.iloc[:,:-1].values
Y_test = test_val1.iloc[:,-1].values
Y_pred = naive_bayes_gaussian(train_val1, X=X_test, Y="is_patient")
accuracy.append(accuracy_score(Y_test, Y_pred))

X_test = test_val2.iloc[:,:-1].values
Y_test = test_val2.iloc[:,-1].values
Y_pred = naive_bayes_gaussian(train_val2, X=X_test, Y="is_patient")
accuracy.append(accuracy_score(Y_test, Y_pred))

X_test = test_val3.iloc[:,:-1].values
Y_test = test_val3.iloc[:,-1].values
Y_pred = naive_bayes_gaussian(train_val3, X=X_test, Y="is_patient")
accuracy.append(accuracy_score(Y_test, Y_pred))

X_test = test_val4.iloc[:,:-1].values
Y_test = test_val4.iloc[:,-1].values
Y_pred = naive_bayes_gaussian(train_val4, X=X_test, Y="is_patient")
accuracy.append(accuracy_score(Y_test, Y_pred))

X_test = test_val5.iloc[:,:-1].values
Y_test = test_val5.iloc[:,-1].values
Y_pred = naive_bayes_gaussian(train_val5, X=X_test, Y="is_patient")
accuracy.append(accuracy_score(Y_test, Y_pred))

print(accuracy)


# In[367]:



def evaluate_algorithm(dataset, n_folds):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
            
        X_test = test_set.iloc[:,:-1].values
        Y_test = test_set.iloc[:,-1].values
        predicted = naive_bayes_gaussian(train_set, test_set, Y="is_patient")
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# In[369]:


print(sum(accuracy)/len(accuracy))


# In[370]:


df


# In[371]:


for col in columns:
    if(col!='is_patient'):
        df[col] = pd.cut(df[col],bins = 5,labels=[1,2,3,4,5])


# In[372]:


len(columns)


# In[373]:


def calculate_prior(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y]==i])/len(df))
    return prior


# In[374]:


def calculate_likelihood_categorical(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    p_x_given_y = len(df[df[feat_name]==feat_val]) / len(df)
    #Laplace correction if the probability is zero
    if p_x_given_y == 0:
        p_x_given_y = (len(df[df[feat_name]==feat_val])+0.5) / (len(df) + 0.5 * len(columns))
    return p_x_given_y


# In[375]:


def naive_bayes_categorical(df, X, Y, cnt=0):
    # get feature names
    features = list(df.columns)[:-1]

    # calculate prior
    prior = calculate_prior(df, Y)

    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_categorical(df, features[i], x[i], Y, labels[j])
            
            if likelihood[j]==0:
                cnt+=1
                
        #print(cnt)
            

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred) 


# In[376]:


train_index = int(0.7 * len(X))
train, test =df[:train_index], df[train_index:]


# In[336]:


X_test = test.iloc[:,:-1].values
Y_test = test.iloc[:,-1].values
Y_pred = naive_bayes_categorical(train, X=X_test, Y="is_patient")
print(accuracy_score(Y_test, Y_pred))


# In[ ]:




