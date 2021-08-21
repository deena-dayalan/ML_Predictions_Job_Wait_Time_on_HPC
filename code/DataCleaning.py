#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="ticks")
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.optimize import curve_fit 
from scipy import stats
import datetime as dt
import statsmodels.api as sm
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from scipy.stats import skew
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
import pickle


# ### Loading raw data into data frame

# In[2]:


##Path to the raw data
#path="/home/d.dasarathan/project_sml/df_raw_data.pkl"
#path ='/home/d.dasarathan/project_sml/100Kdataset/df_raw_100k.pkl'
path ='/home/d.dasarathan/project_sml/100Kdataset/df_raw_600k.pkl'


# In[3]:


df_raw_data=pd.read_pickle(path)
df_raw_data.head()


# In[4]:


df_raw_data.head(-1)


# In[5]:


print('Submit')
print('Min: ',df_raw_data['Submit'].min(axis=0))
print('Max: ',df_raw_data['Submit'].max(axis=0))


# In[6]:


print('End')
print('Min: ',df_raw_data['End'].min(axis=0))
print('Max: ',df_raw_data['End'].max(axis=0))


# In[7]:


print (f"There are total {df_raw_data.shape[0]} jobs")
print ("Total unique users ",len(df_raw_data['User'].unique()))


# In[8]:


df_raw_data.describe()


# In[9]:


df_raw_data.info()


# ### Data Cleaning

# #### Remove the letter 'T' from the timestamp columns

# In[10]:


df_raw_data['Submit']=df_raw_data['Submit'].str.replace('T', ' ')
df_raw_data['End']=df_raw_data['End'].str.replace('T', ' ')
df_raw_data['Start']=df_raw_data['Start'].str.replace('T', ' ')
df_raw_data['Eligible']=df_raw_data['Eligible'].str.replace('T', ' ')


# #### Replace unknown with NaN

# In[11]:


print("Submit column has {} unknown values".format((df_raw_data.Submit == 'Unknown').sum()))
print("Eligible column has {} unknown values".format((df_raw_data.Eligible == 'Unknown').sum()))
print("Start column has {} unknown values".format((df_raw_data.Start == 'Unknown').sum()))
print("End column has {} unknown values".format((df_raw_data.End == 'Unknown').sum()))

print("******Replacing unknown values with NaN******")
df_raw_data['Submit']=df_raw_data["Submit"].replace({"Unknown": np.nan})
df_raw_data['Eligible']=df_raw_data["Eligible"].replace({"Unknown": np.nan})
df_raw_data['Start']=df_raw_data["Start"].replace({"Unknown": np.nan})
df_raw_data["End"]=df_raw_data["End"].replace({"Unknown": np.nan})

print("Submit column has {} unknown values".format((df_raw_data.Submit == 'Unknown').sum()))
print("Eligible column has {} unknown values".format((df_raw_data.Eligible == 'Unknown').sum()))
print("Start column has {} unknown values".format((df_raw_data.Start == 'Unknown').sum()))
print("End column has {} unknown values".format((df_raw_data.End == 'Unknown').sum()))


# #### Drop samples that has invalid or NaN in End Column i.e the job was still runnning at the time of pulling data from SLURM
# #### Also drop samples that has end time later 2021

# In[12]:


df_raw_data.drop(df_raw_data[(df_raw_data['End'].str.contains("NaN")==False) & (df_raw_data['End'].str.contains("20")==False)].index, inplace=True)
df_raw_data.drop(df_raw_data[(df_raw_data['Eligible'].str.contains("NaN")==False) & (df_raw_data['Eligible'].str.contains("20")==False)].index, inplace=True)


# #### Change datatype for timestamp

# In[13]:


df_raw_data['Submit']=pd.to_datetime(df_raw_data['Submit'])
df_raw_data['Start']=pd.to_datetime(df_raw_data['Start'])
df_raw_data['End']=pd.to_datetime(df_raw_data['End'])
df_raw_data['Eligible']=pd.to_datetime(df_raw_data['Eligible'])
df_raw_data.head()


# #### Drop rows where the job started before Submit time (cannot happen in a valid scenario)

# In[14]:


df_raw_data.drop(df_raw_data[df_raw_data.Start < df_raw_data.Submit].index, inplace = True) 


# #### Drop all the records where the job never started or never ended

# In[15]:


df_raw_data.drop(df_raw_data.loc[df_raw_data.Start.isnull()].index, inplace=True)


# In[16]:


# Replacing infinite with nan
df_raw_data.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[17]:


# Reset index
df_cleaned_data=df_raw_data.reset_index(drop=True)


# In[18]:


# Write the cleaned dataset to a pickle file
#df_cleaned_data.to_pickle('/home/d.dasarathan/project_sml/df_cleaned_data_nonulls.pkl')
#df_cleaned_data.to_pickle('/home/d.dasarathan/project_sml/100Kdataset/df_cleaned_data_nonulls.pkl')
df_cleaned_data.to_pickle('/home/d.dasarathan/project_sml/100Kdataset/df_cleaned_data_nonulls_600k.pkl')


# In[ ]:




