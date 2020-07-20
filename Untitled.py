#!/usr/bin/env python
# coding: utf-8

# # To Explore Unsupervised Machine Learning

# From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.

# In[63]:


#import pandas as pd
import pandas as pd


# In[64]:


#read iris dataset
df=pd.read_csv('/Users/saloni/Downloads/Iris.csv')


# In[65]:


#first few rows
print(df.head())


# In[66]:


#to know data types of different columns
df.info()


# In[67]:


#to know max, min, mean, std etc for each column
df.describe()


# In[68]:


#to check whether there are null values in each column
df.isnull().sum()


# # Plotting according to species

# In[69]:


# importing necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt

#pairplot according to different species
sns.set_style("whitegrid")
sns.pairplot(df.drop("Id", axis=1), hue="Species", size=3)
plt.show()
                 


# In[70]:


#setting values of independent variable
X=df.iloc[:,[1,2,3,4]].values


# In[71]:


#setting value of dependent variable
y=df.iloc[:,5].values


# # To know the optimum number of clusters

# In[73]:


# Importing the libraries
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans


# In[74]:


Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(X)
    kmeans.fit(X)
    Error.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1, 11), Error)
plt.title('Optimum number of clusters')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.axvline(3,color="r")
plt.show()


# This is when the within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration.
# From this we choose the number of clusters as ** '3**'.

# # Implement K means clustering with K=3

# In[87]:


# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)


# In[88]:


print(kmeans.cluster_centers_)


# # Visualising Clustering

# In[89]:


# Visualising the clusters - On the first two columns
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# Hence, 3 is the optimum number of clusters.

# In[ ]:




