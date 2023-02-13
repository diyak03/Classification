#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# ### Data cleaning

# Cleaning process: 
# 
# In this cleaning process, I extracted the rows needed to perform my analysis. The original data set was collecting information regarding a specific campaign and since my project is only dealing with customer information it made sense to only extract the columns related to the customer. I then did the following to clean and make the data set easier to work with: 
# 
# - Checked for the unique values in each column to get a better understanding of the data as well as check to see if values need to be changed to be easier to work with 
# - changed certain values to be easier to work with 
# - checked for NA values 
# 

# In[2]:


bank = pd.read_csv("bank.csv")
bank


# In[3]:


# extracting necessary columns 
bank2 = bank[['age', 'job', 'marital', 'education', 'balance', 'default', 'housing', 'loan']]

# checking the unique values for given columns
for col in bank2:
    print(bank2[col].unique())
    
#replacing 'admin.' with 'admin'
bank2 = bank2.replace(['admin.'],'admin')

#checking for NA's 
pd.isnull(bank2).sum() > 0

#dropping rows with negative balance 
bank2 = bank2[bank2['balance'] > 0]

#checking for and removing  duplicates 
bank2 = bank2.drop_duplicates()


#seperating the categorical variables and the continous variables 
cont_var = []
cat_var = []

for v in bank2.columns:
    if bank2[v].dtype == 'int64':
        cont_var += [v]
    else:
        cat_var += [v]
        


# ### Exploratory Data Analysis

# What am I doing?
# To figure out what my data is, i need to explore and play with it to understand it more, this can be attained by looking at the numerical summaries as well as the visualization of the variables. Through this analyis we can gain more insight on the data before asking and answering the questions asked about the data. 
# 
# When looking at the data, here are the questions that I wanted to answer: 
# - what is the distribution of all my variables? of my response variables? does it seem fair and proportiante? in relation to response variable? 
# - are there any outliers? any other significant values that stick out?
# - based on the data, what method would seem best to get the answer I want - "Is there a specific type of person that is more likely to get loans? 
# - can we create a new variable based on customer traits?
# 
# Diya's thoughts: The most fun part of EDA is being able to visualize thousands of data points and although we can't say much about it right of the bat,its still interesting to see and understand your data a little bit better. 

# In[4]:


#checking head and tail of data 
print(bank2.head())
print(bank2.tail())

#understanding the numerical variables of data - age & balance 
print(bank2['age'].describe())
print(bank2['balance'].describe())


# In[5]:



#age 
plt.hist(bank2['age'])
plt.title('Dist. of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

#job
plt.hist(bank2['job'])
plt.title('Dist. of job')
plt.xlabel('Type of Job')
plt.xticks(rotation = 90)
plt.ylabel('Count')
plt.show()

#marital 
plt.hist(bank2['marital'])
plt.title('Dist. of marital status')
plt.xlabel('marital status')
plt.xticks(rotation = 90)
plt.ylabel('Count')
plt.show()

#education
plt.hist(bank2['education'])
plt.title('Dist. of education levels')
plt.xlabel('education levels')
plt.xticks(rotation = 90)
plt.ylabel('Count')
plt.show()

#balance
plt.hist(bank2['balance'])
plt.title('Dist. of bank balance')
plt.xlabel('amount(in dollars)')
plt.ylabel('Count')
plt.show()

#default
d_dat = bank2['default'].value_counts()
labels = 'No', 'Yes'
plt.pie(d_dat,labels= labels)
plt.title('Default')
plt.show()

#housing
h_dat = bank2['housing'].value_counts()
labels = 'No', 'Yes'
plt.pie(h_dat,labels= labels)
plt.title('Customer housing')
plt.show()

#loan
l_dat = bank2['loan'].value_counts()
labels = 'No', 'Yes'
plt.pie(l_dat,labels= labels)
plt.title('Customer loans')
plt.show()


  


# Now that we have seen what our data looks like, let's continue to pre-process it more. Starting of with applying the one hot encoder, then looking at the continous variables and transforming it to get rid of skewness 

# In[13]:


# one hot encoding 
from sklearn.preprocessing import OneHotEncoder

#encoding the categorical variables first 
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(bank2[cat_var]))

# second, we need to add back the indexes
OH_cols.index = bank2.index

# to perform K means we need to just keep the continuous variables
bank2_cont = bank2.drop(cat_var, axis=1)

# add back encoded categoricals to regular continous variables
bank2_OH = pd.concat([bank2_cont, OH_cols], axis=1)


# In[14]:


#lets look at the scatterplot between the two numerical variables
import seaborn as sns
sns.scatterplot(bank2['age'], bank2['balance'])


# In[15]:


# transforming the cont variables 
from sklearn.preprocessing import StandardScaler

bank2_OH[cont_var] = np.log(bank2_OH[cont_var])
scaler = StandardScaler()

bank2_OH[cont_var] = scaler.fit_transform(bank2_OH[cont_var])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8))
sns.distplot(bank2_OH['age'], ax=ax1)
sns.distplot(bank2_OH['balance'], ax=ax2)


# Through EDA it is safe to say the data is clean and workable and now it is time to answer some questions 

# ### Data modelling

# All the data I have in this dataset pertains to a customers attributes, therefore the question arises - are there a certain combination of attributes that makes a customer more likely to get a loan and is there a certain combination of attributes that makes a customer less likely to get a loan? 
# 
# how would this be useful? 
# well if you're a bank you would want to understand who you're customers are, this could be used for many different things - research and development, marketing and general admin. Furthermore, if you look past this specific dataset, this method of making a new group of customers based on variables can help companies make a customer profile so they can cater their products to those customers in a more efficient way. 
# 
# How would i attain my results?
# 
# First I would have to use K means."K-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean". Vector quantization is extremely beneficial for pattern recognition and data compression. Therefore to attain a new "type" of customer using the variables given we would use k means to find a natural grouping. 
# 
# After attaining the groups, i would perform logistic regression since the outcomes are yes or no to see if there is a group that is more likely to get a loan compared to not getting a loan.
# 
# 
# 
# Since balance and age are the only continous numerical data in the data set, we can use that to perform K means

# In[16]:


# elbow method to find optimal amount of clusters 
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):
    clustering = KMeans(n_clusters=i, init='k-means++', random_state=0, n_init=30, max_iter=100)
    clustering.fit(bank2[cont_var])
    wcss.append(clustering.inertia_)
    
plt.plot(range(1,11), wcss, marker='o')
plt.title('Elbow method')
plt.xlabel('k clusters')
plt.ylabel('WCSS')
plt.show()

#from the graph we can see that either 4 or 5 clusters would be optimal 


# In[17]:


# lets see what the clusters look like on our one hot encoded data

kmeans = KMeans(n_clusters=4, init='k-means++', n_init=30, max_iter=100, random_state=0)
clusters = kmeans.fit_predict(bank2_OH)
bank2_OH['cluster'] = clusters
sns.relplot(x='age', y='balance', hue='cluster', data=bank2_OH)

kmeans = KMeans(n_clusters=5, init='k-means++', n_init=30, max_iter=100, random_state=0)
clusters = kmeans.fit_predict(bank2_OH)
bank2_OH['cluster'] = clusters
sns.relplot(x='age', y='balance', hue='cluster', data=bank2_OH)



# In[18]:


# lets use the WCSS method to check the accuracy of the clustering

kmeans4 = KMeans(n_clusters=4)
kmeans4.fit(bank2_OH)
wcss4 = kmeans4.inertia_

kmeans5 = KMeans(n_clusters=5)
kmeans5.fit(bank2_OH)
wcss5 = kmeans5.inertia_

kmeans6 = KMeans(n_clusters=6)
kmeans6.fit(bank2_OH)
wcss6 = kmeans6.inertia_

print(wcss4)
print(wcss5)
print(wcss6)




# In[20]:


# what does this look like on the original scatterplot of age and balance

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0, n_init=30, max_iter=100)
clusters = kmeans.fit_predict(bank2_OH)
bank2['cluster'] = clusters
sns.relplot(x='age', y='balance', hue='cluster', data=bank2)

# you can see the grouping!


# In[21]:


df =  pd.read_csv("bank.csv")
df = df.iloc[:,:8]
df = df[df['balance'] > 0]
df = df.drop_duplicates()

df['cluster'] = clusters
groups = df.groupby(['cluster', 'job', 'marital', 'education', 'default', 'housing', 'loan']).agg(['median', 'sum', 'count']).round()
groups['pct_total'] = (groups['balance']['sum'] / groups['balance']['sum'].sum()).round(3)*100


# In[22]:


top = groups.sort_values(by='pct_total', ascending=False)
top.head(60)


# In[24]:


df.to_excel('final.xlsx', index=False)

