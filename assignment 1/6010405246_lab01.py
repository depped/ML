#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math 


# In[11]:


class neighbors:
    def __init__(self,n_neighbors, p):
        self.n_neighbors = n_neighbors
        self.p = p
    def KNeighborsClassifier(n_neighbors, p) :
        return neighbors( n_neighbors, p) 
    
    def distance(self,d,x):
        sum=0
        for i in range(len(x)):
            sum += (float(d[i])-float(x[i]))**self.p
        return math.sqrt(sum)  
    def vote(self,label):
        most = -1
        count = 0
        for i in label:
            if(label.count(i)>count):
                count = label.count(i)
                most = i
        return most
        
    def fit(self,x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
   
    def predict(self,x):
        nearest_point =[]
        nearest_label = []
        for i in range(self.n_neighbors):
            nearest_point.append(self.distance(self.x_train[i],x[0]))
            nearest_label.append(self.y_train[i])
        for i in range(len(self.x_train)):
            for j in range(self.n_neighbors):
                if(self.distance(self.x_train[i],x[0]) < nearest_point[j]):
                    nearest_point.pop(j)
                    nearest_label.pop(j)
                    nearest_point.append(self.distance(self.x_train[i],x[0]))
                    nearest_label.append(self.y_train[i])
        res = self.vote(nearest_label)
        return [res]
    
    def score(self,x_test,y_test):
        sum = 0       
        for i in range(len(x_test)):
            x = np.array(x_test[i])
            guess = h.predict(x.reshape(1, -1))
            right = y_test[i]
            if(guess == right):
                sum+=1
        return sum/len(x_test)
            


# In[12]:


def train_test_split(x,y,test_size) :
    length = len(x)
    test = int(length*test_size)
    train = length-test
    return x[:train],x[train:],y[:train],y[train:]


# In[13]:


df = pd.read_csv('breast-cancer-wisconsin.csv')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)


# In[14]:


x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[15]:


h = neighbors.KNeighborsClassifier(n_neighbors=7, p=2)
h.fit(x_train, y_train)

new_x = np.array([4,6,5,6,7,8,4,9,1])
result = h.predict(new_x.reshape(1, -1))
print(result)

print(h.score(x_test, y_test))

