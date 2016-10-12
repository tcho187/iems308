
# coding: utf-8

# In[62]:

import pandas as pd
import random as rd

from IPython.display import display
filename='Medicare_Provider_Util_Payment_PUF_CY2014.txt'
n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
s = 10000 #desired sample size
skip = sorted(rd.sample(xrange(1,n+1),n-s))
pd.options.display.max_columns = None
df = pd.read_csv(filename,skiprows=skip,sep='\t')


# In[63]:

df=df.dropna() #Dropping rows with any columns with NA values


# In[64]:

print df.shape


# In[65]:

print df.iloc[0] #prints out columns and first row value. Good for visually seeing variable types


# In[66]:

print df.columns.values # a list of variables


# In[67]:

pd.set_option('expand_frame_repr', True)
print df.describe() #statistics


# In[68]:

print df.corr() #correlation


# In[69]:

get_ipython().magic(u'matplotlib inline')
df.hist(figsize=(14,14))


# In[89]:

df['average_Medicare_allowed_amt'].hist()


# In[70]:

import numpy as np
df['submitted_minus_allowed']=df['average_submitted_chrg_amt']-df['average_Medicare_allowed_amt']
# print df[['average_submitted_chrg_amt','average_Medicare_allowed_amt','submitted_minus_allowed']]
df['submitted_minus_allowed']+=0.5
log8=np.log(df['submitted_minus_allowed'])
df['log_submitted_minus_allowed']=log8
df['log_submitted_minus_allowed'].hist()


# In[90]:

log1 = np.log(df['average_Medicare_allowed_amt']+0.5)
df['log_average_Medicare_allowed_amt']=log1
df['log_average_Medicare_allowed_amt'].hist()
log2 = np.log(df['average_Medicare_payment_amt']+0.5)
df['log_average_Medicare_payment_amt']=log2

log3 = np.log(df['average_Medicare_standard_amt']+0.5)
df['log_average_Medicare_standard_amt']=log3

log4 = np.log(df['average_submitted_chrg_amt']+0.5)
df['log_average_submitted_chrg_amt']=log4

log5 = np.divide(1,np.log(df['bene_unique_cnt']+0.5))
df['log_bene_unique_cnt']=log5

log6=np.divide(1,np.log(df['bene_day_srvc_cnt']+0.5))
df['log_bene_day_srvc_cnt']=log6

log7=np.divide(1,np.log(df['line_srvc_cnt']+0.5))
df['log_line_srvc_cnt']=log7


# In[72]:

print df['log_bene_day_srvc_cnt'].hist()


# In[73]:

regions= {'FL':'south','KY':'south','GA':'south','MD':'south','AL':'south','LA':'south' ,'OK':'south'          ,'TX':'south','AR':'south','MS':'south','TN':'south','SC':'south','NC':'south','WV':'south'          ,'VA':'south','DC':'south','DE':'south'         ,'NJ':'northeast','PA':'northeast','NY':'northeast','CT':'northeast','RI':'northeast'          ,'MA':'northeast','VT':'northeast','NH':'northeast','ME':'northeast'         ,'ND':'midwest','SD':'midwest','NE':'midwest','KS':'midwest','MN':'midwest'          ,'IA':'midwest','MO':'midwest','WI':'midwest','IL':'midwest','IN':'midwest','MI':'midwest','OH':'midwest'         ,'WA':'west','OR':'west','CA':'west','NV':'west','ID':'west'          ,'MT':'west','WY':'west','UT':'west','AZ':'west','CO':'west','NM':'west'         ,'HI':'pacific','AK':'pacific'         ,'PR':'puerto_rico'}
specific_regions= {'FL':'south_atlantic','KY':'east_south','GA':'south_atlantic','MD':'south_atlantic','AL':'east_south','LA':'west_south' ,'OK':'west_south'          ,'TX':'west_south','AR':'west_south','MS':'east_south','TN':'east_south','SC':'south_atlantic','NC':'south_atlantic','WV':'south_atlantic'          ,'VA':'south_atlantic','DC':'south_atlantic','DE':'south_atlantic'         ,'NJ':'middle_atlantic','PA':'middle_atlantic','NY':'middle_atlantic','CT':'new_england','RI':'new_england'          ,'MA':'new_england','VT':'new_england','NH':'new_england','ME':'new_england'         ,'ND':'west_central','SD':'west_central','NE':'west_central','KS':'west_central','MN':'west_central'          ,'IA':'west_central','MO':'west_central','WI':'east_central','IL':'east_central','IN':'east_central','MI':'east_central','OH':'east_central'         ,'WA':'pacific','OR':'pacific','CA':'pacific','NV':'mountain','ID':'mountain'          ,'MT':'mountain','WY':'mountain','UT':'mountain','AZ':'mountain','CO':'mountain','NM':'mountain'         ,'HI':'pacific','AK':'pacific'         ,'PR':'puerto_rico'}
df['regions'] = df['nppes_provider_state'].map(regions)
df['specific_regions']=df['nppes_provider_state'].map(specific_regions)


# In[74]:

features=['log_submitted_minus_allowed','place_of_service','nppes_provider_gender','hcpcs_drug_indicator' ]
df2=pd.get_dummies(df[features])
print df2.shape
count=list(range(df2.shape[0]))
labels=df2.columns.values
print labels


# In[75]:

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing 

stdsc = StandardScaler()
scaled = stdsc.fit_transform(df2)
print(scaled)


# In[76]:

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
distortions = []
for i in range(1,20):
    km = KMeans(n_clusters=i,init='k-means++', n_init=10,max_iter=300,random_state=0)
    km.fit(scaled)
    distortions.append(km.inertia_)
plt.plot(range(1,20),distortions, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


# In[77]:

km = KMeans(n_clusters=4,init='k-means++',n_init=10,max_iter=300,tol=1e-04,random_state=0)
y_km=km.fit_predict(scaled)
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(scaled,y_km,metric='euclidean')
y_ax_lower, y_ax_upper = 0,0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i/n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),c_silhouette_vals,height=1.0,edgecolor='none',color=color)
    yticks.append((y_ax_lower +y_ax_upper)/2)
    y_ax_lower +=len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,color="red",linestyle="--")
plt.yticks(yticks, cluster_labels+1)
plt.title('Silhouette plot')
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()


# In[78]:

df['labels']=y_km
pd.set_option('expand_frame_repr', False)
# print(df[['regions','log_average_Medicare_payment_amt','log_average_Medicare_allowed_amt' , 'labels']])


# In[79]:

pd.set_option('expand_frame_repr', True)
print pd.crosstab(df['labels'],df['place_of_service'])


# In[80]:

print pd.crosstab(df['labels'],df['nppes_provider_gender'])


# In[81]:

print pd.crosstab(df['labels'],df['hcpcs_drug_indicator'])


# In[82]:

print df['labels'].groupby(df['labels']).count()


# In[85]:

print pd.crosstab(df['labels'],df['regions'])


# In[88]:

pd.set_option('expand_frame_repr', True)
print pd.crosstab(df['labels'],df['specific_regions'])


# In[83]:

pd.set_option('expand_frame_repr', False)
df.groupby('labels').mean()


# In[84]:

print df['submitted_minus_allowed'].groupby(df['labels']).mean()


# In[ ]:



