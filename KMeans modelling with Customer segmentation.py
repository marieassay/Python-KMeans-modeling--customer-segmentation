#!/usr/bin/env python
# coding: utf-8
# In[1]:

#Import libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Read in csv data

df = pd.read_csv(r'C:\Users\Ibrahim Nas\Downloads\Mall_Customers.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# # Univariate Analysis 

# In[5]:


sns.displot(df['Annual Income (k$)'])


# In[6]:


df.columns


# In[7]:


columns=['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.displot(df[i]);


# In[8]:


sns.displot(df, x="Annual Income (k$)", hue="Gender", kde=True);


# In[9]:


columns=['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.displot(df, x=i, hue="Gender", kde=True);


# In[10]:


columns=['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']
for i in columns:
    plt.figure()
sns.boxplot(data=df,x='Gender', y=df[i]);


# In[11]:


df['Gender'].value_counts(normalize=True)


# # Bivariate Analysis

# In[12]:


#Create a scatterplot using 2 variables- Annual Income and Spending Score
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)')


# In[13]:


# Create pairplot for all variables
sns.pairplot(df)


# In[14]:


# Specify to remove/drop the CustomerID column along the columns axis by using axis=1
df = df.drop('CustomerID', axis=1)
sns.pairplot(df)


# In[15]:


#Create an additional hyperparameter on the pairplot to examine the insights between Male and Females cohorts
sns.pairplot(df, hue='Gender')


# In[16]:


# Finding the mean values of 3 variable columns Age, Annual Income,Spending Score
df.groupby(['Gender'])['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# In[17]:


# Finding the correlation between the 3 variables
df.corr()


# In[18]:


sns.heatmap(df.corr(),annot=True,cmap='coolwarm')


# # KMeans Clustering- Univariate, Bivariate, Multivariate

# In[19]:


#Initiating the machine learning algorithm and then fitting the data then define labels out of the model
#Univariate clustering
clustering1 = KMeans(n_clusters=3)


# In[20]:


#Fit the data(Annual Income column) into the model
clustering1.fit(df[['Annual Income (k$)']])
KMeans()


# In[21]:


clustering1.labels_


# In[22]:


#Defining the labels with the data(Annual Income column) and viewing a snippet of the output
df['Income Cluster']= clustering1.labels_
df.head()


# In[23]:


#Checking how many customers fall within the 6 different Income clusters
df['Income Cluster'].value_counts()


# In[24]:


clustering1.inertia_


# In[25]:


#Initiating a for loop to re-define the number of clusters for our data    
inertia_scores = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(kmeans.inertia_)


# In[26]:


#Plotting an elbow curve, to determine the optimal number of clusters for the KMeans model
plt.plot(range(1,11),inertia_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia Score')
plt.title('Elbow Curve')
plt.show()


# In[27]:


df.groupby('Income Cluster')['Age', 'Annual Income (k$)', 'Spending Score (1-100)'].mean()


# In[28]:


#Bivariate Clustering Model defined with 5 clusters
clustering2 = KMeans(n_clusters=5)
clustering2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
df['Income and Spending Cluster'] = clustering2.labels_
df.head()


# In[29]:


#KMeans model for bivariate model using a for loop to re-define the number of clusters
inertia_scores2 = []
for i in range(1,11):
    kmeans2 = KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    inertia_scores2.append(kmeans2.inertia_)


# In[30]:


#Plotting an elbow curve, to find the optimal number of clusters for this second Bivariate KMeans model.
plt.plot(range(1,11), inertia_scores2)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia Score')
plt.title('Elbow Curve2')
plt.show()


# In[31]:


#Creating a dataframe from the 5 clusters defined in the bivariate KMeans model
centers=pd.DataFrame(clustering2.cluster_centers_)
centers.columns=['x','y']
centers


# In[32]:


#Visualizing the above bivariate elbow curve in a scatterplot with marked cluster centers
plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'], y=centers['y'], s=100, c='black', marker='*')
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Income and Spending Cluster', palette='tab10')

plt.savefig('bivariate_clustering_centers.png')


# In[33]:


#Cross-tabulation table showing the proportion of individuals in each gender category for income and spending clusters
pd.crosstab(df['Income and Spending Cluster'], df['Gender'], normalize='index')


# In[34]:


#Identifying patterns or trends by calculating the average age of each of the 5 segmentated clusters
df.groupby('Income and Spending Cluster')['Age', 'Annual Income (k$)', 'Spending Score (1-100)'].mean()


# In[35]:


#Multivariate clustering
#Importing scaler to scale data
from sklearn.preprocessing import StandardScaler


# In[36]:


Scaler = StandardScaler()


# In[37]:


#Creating binary variables for the categorical column- Gender and the data and viewing a sample of the result
dff = pd.get_dummies(df)
dff.head()


# In[38]:


#Dropped one of the columns for Gender and merged the Gender data
dff = pd.get_dummies(df, drop_first=True)
dff.head()


# In[39]:


#Renaming the Gender column
dff = dff.rename(columns={'Gender_Male': 'Gender: F=0/M=1'})
dff.head()


# In[40]:


dff.columns


# In[41]:


dff=dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender: F=0/M=1']]
dff.head()


# In[42]:


#Scaling the data and creating an array
dff = Scaler.fit_transform(dff)


# In[43]:


#Creating a new DataFrame using the resulting array
dff = pd.DataFrame(Scaler.fit_transform(dff))
dff.head()


# In[44]:


#KMeans model for multivariate model using a for loop to re-define the number of clusters
inertia_scores3=[]
for i in range(1,11):
    kmeans3=KMeans(n_clusters=i)
    kmeans3.fit(dff)
    inertia_scores3.append(kmeans3.inertia_)


# In[45]:


#Plotting an elbow curve, to find the optimal number of clusters for this Multivariate KMeans model.
plt.plot(range(1,11), inertia_scores3)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia Score')
plt.title('Elbow Curve3')
plt.show()


# In[46]:


#Multivariate Clustering Model defined with 4 clusters
clustering3 = KMeans(n_clusters=4)
clustering3.fit(dff)
df['Income and Spending Cluster'] = clustering3.labels_
df.head()


# In[47]:


#Aware 'CustomerID' column will not be in dff, so adding the clumn back in for a unique identifier as needed
df['CustomerID'] = pd.Series(range(1, len(df)+1))
id_column = df['CustomerID']    #Getting the id column from the original data frame
dff.insert(0, 'CustomerID', id_column)    #Adding the id column to the front of this dff dataframe


# In[48]:


df


# In[49]:


df.to_csv('clustering.csv')


# In[ ]:




