#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("Train_B_Tree.csv")


# In[3]:


df = df.sample(frac=1).reset_index()


# In[4]:


df.isnull().sum()


# In[5]:


feature_name=df.columns


# In[6]:


df.dtypes


# In[7]:


train_index = int(0.7 * len(df))
train, test =df[:train_index], df[train_index:]


# In[8]:


class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        
        # for leaf node
        self.value = value


# In[9]:


maxdepth = 0


# In[10]:


class DecisionTreeRegressor():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''
        #if curr_depth > maxdepth:
        #   maxdepth = curr_depth
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        best_split = {}
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["var_red"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["var_red"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_var_red = -float("inf")
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    # update the best split if needed
                    if curr_var_red>max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def variance_reduction(self, parent, l_child, r_child):
        ''' function to compute variance reduction '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction
    
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        val = np.mean(Y)
        return val
                
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print(str(feature_name[tree.feature_index]), "<=", tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def tree_depth(self, tree=None):
        if not tree:
            tree = self.root
        if(tree.value is not None):
            return 0
        
        else:
            return 1+max(self.tree_depth(tree.left),self.tree_depth(tree.right))
        
        
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
        
    def make_prediction(self, x, tree):
        ''' function to predict new dataset '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    
    def predict(self, X):
        ''' function to predict a single data point '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions


# In[11]:


X_train, X_test, y_train, y_test = train.iloc[:,:-1].values, test.iloc[:,:-1].values,  train.iloc[:,-1].values.reshape(-1,1), test.iloc[:,-1].values.reshape(-1,1)


# In[12]:


print(y_train)


# In[13]:


regressor = DecisionTreeRegressor(min_samples_split=3, max_depth=5)
regressor.fit(X_train,y_train)
regressor.print_tree()
regressor.tree_depth()


# In[14]:


def error_term(Y_pred,y_test): 
    Y_pred = regressor.predict(X_test)
    sum=0
    for i in range(len(Y_pred)):
        sum+=((y_test[i] - Y_pred[i])**2)

    return sum/len(y_test)


# In[18]:


import random
best_split = 0
error = float("inf")
depth = 0
for i in range(10):
    splits = random.randint(1,10)
    regressor = DecisionTreeRegressor(min_samples_split=splits, max_depth=4)
    regressor.fit(X_train,y_train)
    Y_pred = regressor.predict(X_test) 
    error1 = error_term(Y_pred, y_test)
    if(error1 < error):
        error = error1
        best_split = splits
        depth = regressor.tree_depth()


# In[19]:


print("Best Decision tree is with depth %d,error %d and min_split %d" %(depth,error, best_split))


# In[20]:


regressor = DecisionTreeRegressor(min_samples_split=best_split, max_depth=depth)
regressor.fit(X_train,y_train)
regressor.print_tree()
regressor.tree_depth()


# In[27]:


depths = [2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20]
error = []
x = float("inf")
for i in depths:
    regressor = DecisionTreeRegressor(min_samples_split=i, max_depth=i)
    regressor.fit(X_train,y_train)
    Y_pred = regressor.predict(X_test)
    error.append(error_term(Y_pred,y_test))
    error1 = error_term(Y_pred,y_test)
    if(error1<x):
        x = error1
        overDepth = i


# In[28]:


error


# In[29]:


plt.plot(depths, error)
plt.show()


# In[24]:


overFittedDepth = overDepth


# In[25]:


print(overFittedDepth)


# In[31]:


regressor = DecisionTreeRegressor(min_samples_split=4, max_depth=13)
regressor.fit(X_train,y_train)
regressor.print_tree()
regressor.tree_depth()


# In[ ]:




