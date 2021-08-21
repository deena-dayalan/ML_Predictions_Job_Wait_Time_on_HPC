import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
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


# In[2]:


#df_main = pd.read_pickle('/home/d.dasarathan/project_sml/df_cleaned_data_nonulls.pkl')
#df_main = pd.read_pickle('/home/d.dasarathan/project_sml/100Kdataset/df_cleaned_data_nonulls.pkl')
df_main = pd.read_pickle('/home/d.dasarathan/project_sml/100Kdataset/df_cleaned_data_nonulls_600k.pkl')


# #### Calculate job wait time (Target variable) based on when the job was ready to be put in queue

# In[3]:


df_main['waitTime'] = df_main.Start - df_main.Submit
df_main['waitTime']


# In[4]:


df_main['waitTimeHr'] = df_main.waitTime / np.timedelta64(1, 'h')
df_main['waitTimeHr']


# In[5]:


# Extract int value and char unit from ReqMem column
df_main[['ReqMem_INT', 'ReqMem_unit']] = df_main['ReqMem'].str.extract('(\d+\.?\d*)([A-Za-z]*)', expand = True)


# In[6]:


df_main.ReqMem_INT = df_main.ReqMem_INT.astype(float)
#df_main.ReqNodes = df_main.ReqNodes.astype(int)
df_main.ReqNodes = df_main.ReqNodes.astype(str)
df_main.ReqNodes = df_main.ReqNodes.str.replace(r'\D', '').astype(int)
df_main.ReqCPUS = df_main.ReqCPUS.astype(int)
df_main.NCPUS = df_main.NCPUS.astype(int)


# In[7]:


# Create a new column that has total mem requested in a job
df_main['Req_totalMem'] = pd.Series()


# In[8]:


df_main['Req_totalMem'] = np.where (df_main.ReqMem_unit == 'Gn', (df_main.ReqMem_INT * df_main.ReqNodes),df_main['Req_totalMem'])
df_main['Req_totalMem'] = np.where (df_main.ReqMem_unit == 'Gc' , (df_main.ReqMem_INT * df_main.ReqCPUS), df_main['Req_totalMem'])
df_main['Req_totalMem'] = np.where (df_main.ReqMem_unit == 'Mc' , ((df_main.ReqMem_INT * df_main.ReqCPUS)/1024) , df_main['Req_totalMem'])
df_main['Req_totalMem'] = np.where (df_main.ReqMem_unit == 'Mn', ((df_main.ReqMem_INT * df_main.ReqNodes)/1024) ,df_main['Req_totalMem'])
df_main['Req_totalMem'] = np.where (df_main.ReqMem_unit == 'n' , 0 , df_main['Req_totalMem'])
df_main['Req_totalMem'] = np.where (df_main.ReqMem_unit == 'Tn', (df_main.ReqMem_INT * df_main.ReqNodes * 1024) ,df_main['Req_totalMem'])
df_main['Req_totalMem'] = np.where (df_main.ReqMem_unit == 'Tc' , (df_main.ReqMem_INT * df_main.ReqCPUS * 1024), df_main['Req_totalMem'])


# In[9]:


# Split days and duration into two columns and then create a new measure that has hours (sum of both columns)
df_main['TLdays']=np.where(df_main.Timelimit.str.contains("-"),df_main['Timelimit'].str.split('-').str[0],'0')

# Change the data type from str to int
df_main['TLdays']=pd.to_numeric(df_main.TLdays)

# Replace "Partition Limit" & "UNLIMITED" values in the time limit column with "23:59:59"
df_main['TLduration']=np.where(df_main.Timelimit.str.contains("-"),df_main['Timelimit'].str.split('-').str[1],df_main['Timelimit'])
df_main.TLduration=np.where(df_main.TLduration.str.contains('Partition_Limit'), "23:59:59", df_main.TLduration)
df_main.TLduration=np.where(df_main.TLduration.str.contains('UNLIMITED'), "23:59:59", df_main.TLduration)

# Change the data type from string to time
df_main.TLduration=pd.to_timedelta(df_main.TLduration)


# In[10]:


# Calculate one field from TLdays & TLduration and drop the rest (not required) 
df_main['Timelimit_hr'] = (df_main.TLdays * 24) + (df_main.TLduration / np.timedelta64(1, 'h'))


# In[11]:


# Change the value of partition where it contains a "," Example: 'short,large,express,west', 'short,reservation', 'short,express'
df_main.Partition=np.where(df_main.Partition.str.contains(','), "partcombo", df_main.Partition)


# In[12]:


# Drop unnecessary columns (derived new features using these)
df_main=df_main.drop(['ReqMem','ReqMem_INT','ReqMem_unit','TLdays','TLduration','Timelimit'], axis=1)


# In[13]:


# Fill the nan values in priority column with mean priority so that the input does not introduce bias.
df_main['Priority']=df_main['Priority'].fillna((df_main['Priority'].mean()))


# In[14]:


# Creating a new column using submit time of the job. The new column tells us in which quarter of the day the job was submitted
def getquartofday(dtcol):
    QODcol = dtcol.dt.strftime("%H")
    QODcol = np.where((QODcol == '00') | (QODcol == '01') | (QODcol == '02') | (QODcol == '03') | (QODcol == '04') | (QODcol == '05'), 'q1', QODcol)
    QODcol = np.where((QODcol == '06') | (QODcol == '07') | (QODcol == '08') | (QODcol == '09') | (QODcol == '10') | (QODcol == '11'), 'q2', QODcol)
    QODcol = np.where((QODcol == '12') | (QODcol == '13') | (QODcol == '14') | (QODcol == '15') | (QODcol == '16') | (QODcol == '17'), 'q3', QODcol)
    QODcol = np.where((QODcol == '18') | (QODcol == '19') | (QODcol == '20') | (QODcol == '21') | (QODcol == '22') | (QODcol == '23'), 'q4', QODcol)
    return QODcol

df_main['QOD'] = getquartofday(df_main.Submit)


# In[15]:


# Creating a new column using submit time of the job. The new column tells us in which quarter of the year the job was submitted
def getquartofyear(dtcol):
    QOYcol = dtcol.dt.strftime("%b")
    QOYcol = np.where((QOYcol == 'Jan') | (QOYcol == 'Feb') | (QOYcol == 'Mar'), 'q1', QOYcol)
    QOYcol = np.where((QOYcol == 'Apr') | (QOYcol == 'May') | (QOYcol == 'Jun'), 'q2', QOYcol)
    QOYcol = np.where((QOYcol == 'Jul') | (QOYcol == 'Aug') | (QOYcol == 'Sep'), 'q3', QOYcol)
    QOYcol = np.where((QOYcol == 'Oct') | (QOYcol == 'Nov') | (QOYcol == 'Dec'), 'q4', QOYcol)
    return QOYcol

df_main['QOY'] = getquartofyear(df_main.Submit)


# In[16]:


# Create exlusive flag just like the slurm parameter
df_main['exclusive']=np.where(df_main.NCPUS != df_main.ReqCPUS, 1, 0)


# In[17]:


# Calculate total core hours
df_main['corehrs']= df_main.Timelimit_hr * df_main.ReqCPUS


# In[18]:


#df_main_req = df_main[['ReqNodes', 'Start', 'End', 'Submit', 'Priority', 'Partition', 'waitTimeHr', 'Req_totalMem', 'corehrs', 'QOD', 'QOY', 'exclusive']]
#df_main_req.head()


# In[19]:


def calc_rnjobct(rowsubmit, start, end):
    return len(np.where((rowsubmit > start) & (rowsubmit < end))[0])

# Make pandas apply function faster using swifter
# https://stackoverflow.com/questions/45545110/make-pandas-dataframe-apply-use-all-#cores
# data['out'] = data['in'].swifter.apply(some_function)
# !pip install swifter
# import swifter
# In[ ]:


# Testing on full dataset
import swifter
import time
start_time = time.time()

#df_main_req['rnjbct'] = df_main_req[['Submit','Start','End']].swifter.apply(lambda row: calc_rnjobct(row['Submit'], df_main_req.Start, df_main_req.End), axis=1)
df_main['rnjbct'] = df_main[['Submit','Start','End']].swifter.apply(lambda row: calc_rnjobct(row['Submit'], df_main.Start, df_main.End), axis=1)

print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:


#df_main_req.to_pickle('/scratch/d.dasarathan/df_main1_first4M.pkl')
#df_main_req.head()
df_main.head()


# In[ ]:


# Reset index
#df_main_req=df_main_req.reset_index(drop=True)
#df_main_req.head()
df_main=df_main.reset_index(drop=True)
df_main.head()


# In[ ]:


# Write the cleaned dataset to a pickle file
#df_main_req.to_pickle('/scratch/d.dasarathan/df_newdata_cleaned_added_features.pkl')
#df_main.to_pickle('/scratch/d.dasarathan/df_newdata_cleaned_100k_added_features.pkl')
df_main.to_pickle('/scratch/d.dasarathan/df_newdata_cleaned_600k_added_features.pkl')


# In[ ]:


#df_sbatch = pd.read_pickle('/scratch/d.dasarathan/df_newdata_cleaned_added_features_sbatch.pkl')
#df_test = pd.read_pickle('/scratch/d.dasarathan/df_newdata_cleaned_100k_added_features.pkl')
df_test = pd.read_pickle('/scratch/d.dasarathan/df_newdata_cleaned_600k_added_features.pkl')


# In[ ]:


df_test.head()


# In[ ]:


df_test.shape


# In[ ]:


df_test.describe()


# In[ ]:




