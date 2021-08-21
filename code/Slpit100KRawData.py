#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle


# In[2]:


df_full = pd.read_pickle('/home/d.dasarathan/project_sml/df_raw_data.pkl')


# In[ ]:


#df_full = df_full.sample(frac=1).reset_index(drop=True)
#df_full.head()


# In[ ]:


#df_full.shape


# In[ ]:


print('Submit')
print('Min: ',df_full['Submit'].min(axis=0))
print('Max: ',df_full['Submit'].max(axis=0))


# In[ ]:


print('End')
print('Min: ',df_full['End'].min(axis=0))
print('Max: ',df_full['End'].max(axis=0))


# In[ ]:


df_100k = df_full.head(600000)


# In[ ]:


#df_100k.shape


# In[ ]:


df_100k.to_pickle('/home/d.dasarathan/project_sml/100Kdataset/df_raw_600k.pkl')


# In[ ]:


df_100k_load = pd.read_pickle('/home/d.dasarathan/project_sml/100Kdataset/df_raw_600k.pkl')
#df_100k_load.shape


# In[ ]:


df_100k_load.head()

