
# coding: utf-8

# # Introduction
# In this problem,We have to train a model to identify unique patients in the sample dataset
# given.We are given with a csv file which contains Name,gender,DOB and father’s name.As data
# can come from various sources,there is great possiblity of data redundancy.So we have to train
# our model to remove that redundancy by checking out those datas which are similar and be-
# longs to same person.So for solving this problem,I have used PYTHON and used packages like
# numpy,pandas and sklearn.

# In[31]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import time
import datetime
from sklearn import preprocessing, cross_validation, svm, metrics
from sklearn.cluster import KMeans

df = pd.read_csv("data/train.csv")
df.head()


# # Data Preprocessing:
# So,as given data is in very raw form,for applying algorithm,we have to convert it into suitable
# form.So here we are reading the data from given csv file -train.csv which is present in data sub-
# folderAs we have to convert data into suitable datatype,we are counting the number of letters
# 1in fathers name (fn) and then will be sorting the data in descending order of name.(fn).To make
# cluster of people with simillar fathers name,we will assign a lable to fn_len,and we will change
# them for people with different fathers name.To differentiate people with different first names,we
# will do same thing on (ln) and get (ln_lable). To convert gender into suitable datatype,we will be
# assigning 0 to male and 1 to female. To convert DOB into suitbale datatype,we will be using unix
# date format that has unique integer assigned to every day in any year.

# In[32]:


df['fn_len'] = df['fn'].apply(lambda val:len(val))


# In[33]:


df.head()


# In[34]:


df.sort_values(['fn'],ascending=False,inplace=True)
df.reset_index(inplace=True)
df.drop(['index'],1,inplace=True)


# In[35]:


j = 0
df['fn_label'] = df['fn_len']
df['fn_label'][0] = 0
for i in range(1,len(df)):
    if(df['fn'][i] in df['fn'][i-1]):
        df['fn_label'][i] = df['fn_label'][i-1]
    else:
        j += 1
        df['fn_label'][i] = j


# In[36]:


df.head()


# In[37]:


df['ln_len'] = df['ln'].apply(lambda val:len(val))
df.sort_values(['ln'],ascending=False,inplace=True)
df.reset_index(inplace=True)
df.drop(['index'],1,inplace=True)

j = 0
df['ln_label'] = df['ln_len']
df['ln_label'][0] = 0
for i in range(1,len(df)):
    if(df['ln'][i] in df['ln'][i-1]):
        df['ln_label'][i] = df['ln_label'][i-1]
    else:
        j += 1
        df['ln_label'][i] = j


# In[38]:


df.head()


# In[39]:


df["val_gn"] = df["gn"].map({"M":0,"F":1})


# In[40]:


def DOB_to_unix(str):
    s = str.split('/')
    str = s[0] + '/' + s[1]+ '/19' + s[2]
    
    return int((time.mktime(datetime.datetime.strptime(str, "%d/%m/%Y").timetuple())))

df["val_dob"] = df["dob"].apply(lambda x: DOB_to_unix(x))


# In[41]:


df.head()


# In[42]:


train = np.array(df[['fn_label','ln_label','val_gn','val_dob']])
scaler = preprocessing.StandardScaler().fit(train)
train = scaler.transform(train)
train.shape


# # Learning Algorithm:
# Here in this problem,I will be using K means clustering algo.Initially we will be forming n number
# of clusters which are equal to number of different elements in training set -train.csv.We will give
# training points to fit() function and will make groups using predict() function.55 (printed o/p)is
# number of different final clusters remained after deduplication.After sorting,we will drop the rows
# having duplicate values and will only take one value per cluster in our final type.csv file.

# In[43]:



n_cluster = len(train)
kmn = KMeans(n_clusters=n_cluster)
kmn.fit(train)
groups = kmn.predict(train)
out = df
out['type'] = groups
print(len(out['type'].unique()))


# In[44]:


out.sort_values(['type'],ascending=False,inplace=True)
out.drop_duplicates('type')
out.reset_index(inplace=True)
out.drop(['index'],1,inplace=True)
out.head()


# In[45]:


out.reset_index(inplace=True)
out.drop(['index'],1,inplace=True)
out.head()
out = df[['ln','dob','gn','fn']]


# In[46]:


out.to_csv('type.csv',index=None)

