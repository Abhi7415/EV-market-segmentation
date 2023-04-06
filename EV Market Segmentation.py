#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


dfev = pd.read_csv(r"C:/Users/Abhinav/Desktop/Global EVdata.csv")
dfev.head()


# In[3]:


dfev.drop('Unnamed: 0', axis=1, inplace=True)


# In[4]:


dfev.head(2)


# In[6]:


dfev['price_INR'] = dfev['PriceEuro']*87.219


# In[7]:


dfev.head()


# In[8]:


dfev['RapidCharge'].replace(to_replace=['Yes','No'],value=[1,0],inplace=True)


# In[9]:


dfev.head()


# In[10]:


dfev.shape


# In[11]:


dfev.isnull().sum()


# In[12]:


dfev.info()


# In[13]:


dfev['Brand'].value_counts()


# In[14]:


dfev['Brand'].value_counts().sum()


# In[16]:


plt.figure(figsize=(12,8))
dfev['Brand'].value_counts().plot(kind='bar')
plt.grid()


# In[18]:


pip install plotly


# In[19]:


import plotly.express as px
import plotly.io as pio


# In[20]:


fig = px.bar(dfev,x='Brand',y = 'TopSpeed_KmH',color = 'Brand',title = 'Which Car Has a Top speed?',labels = {'x':'Car Brands','y':'Top Speed Km/H'})
pio.show(fig)


# In[21]:


fig = px.bar(dfev,x='AccelSec',y = 'Brand',color = 'Brand',title = 'Which car has fastest accelaration?',labels = {'x':'Accelaration','y':'Car Brands'})
pio.show(fig)


# In[22]:


dfev['price_INR'].plot(figsize = (12,8),title='Car Price',xlabel = 'No. of Samples',ylabel = 'Car Price',color = 'green')


# In[23]:


fig = px.bar(dfev,x = 'Range_Km',y = 'PowerTrain',color = 'PowerTrain',text='PowerTrain')
pio.show(fig)


# In[24]:


fig = px.pie(dfev,names = 'Brand',values = 'price_INR')
pio.show(fig)


# In[25]:


fig = px.bar(dfev,x = 'Brand',y = 'price_INR',color = 'Brand')
pio.show(fig)


# In[26]:


dfev.head()


# In[27]:


fig = px.scatter_3d(dfev,x = 'Brand',y = 'Seats',z = 'Segment',color='Brand')
pio.show(fig)


# In[28]:


fig = px.scatter_3d(dfev,x = 'Brand',y = 'AccelSec',z = 'price_INR',color = 'Brand')
pio.show(fig)


# In[29]:


fig = px.box(dfev,x='RapidCharge',y = 'price_INR',color = 'RapidCharge',points='all')
pio.show(fig)


# In[30]:


fig = px.scatter(dfev,x = 'Brand',y = 'Range_Km',size='Seats',color = 'Brand',hover_data=['RapidCharge','price_INR'])
pio.show(fig)


# In[31]:


## Pairplot of all the columns based on Rapid Charger presence

sns.pairplot(dfev,hue='RapidCharge')
plt.show()


# In[32]:


## Heatmap to show the correlation of the data

ax= plt.figure(figsize=(15,8))
sns.heatmap(dfev.corr(),linewidths=1,linecolor='white',annot=True)


# In[33]:


a=np.arange(1,104)
a


# In[35]:


# Making Some Comparisions for our dataset

## Frequency of the Brands in the dataset

ax= plt.figure(figsize=(20,5))
sns.barplot(x='Brand',y=a,data=dfev)
plt.grid(axis='y')
plt.title('Brands in the datset')
plt.xlabel('Brand')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()
print("Byton , Fiat and smart are the prominent brands and Polestar being the least")


# In[36]:


## Top speeds achieved by the cars of a brand

ax= plt.figure(figsize=(20,5))
sns.barplot(x='Brand',y='TopSpeed_KmH',data=dfev,palette='Paired')
plt.grid(axis='y')
plt.title('Top Speed achieved by a brand')
plt.xlabel('Brand')
plt.ylabel('Top Speed')
plt.xticks(rotation=45)
plt.show()
print("Porsche, Lucid and Tesla produce the fastest cars and Smart the lowest")


# In[37]:


## Range a car can achieve

ax= plt.figure(figsize=(20,5))
sns.barplot(x='Brand',y='Range_Km',data=dfev,palette='tab10')
plt.grid(axis='y')
plt.title('Maximum Range achieved by a brand')
plt.xlabel('Brand')
plt.ylabel('Range')
plt.xticks(rotation=45)
plt.show()
print('Lucid, Lightyear and Tesla have the highest range and Smart the lowest')


# In[38]:


## Car efficiency

ax= plt.figure(figsize=(20,5))
sns.barplot(x='Brand',y='Efficiency_WhKm',data=dfev,palette='hls')
plt.grid(axis='y')
plt.title('Efficiency achieved by a brand')
plt.xlabel('Brand')
plt.ylabel('Efficiency')
plt.xticks(rotation=45)
plt.show()
print("Byton , Jaguar and Audi are the most efficient and Lightyear the least")


# In[39]:


## Number of seats in each car

ax= plt.figure(figsize=(20,5))
sns.barplot(x='Brand',y='Seats',data=dfev,palette='husl')
plt.grid(axis='y')
plt.title('Seats in a car')
plt.xlabel('Brand')
plt.ylabel('Seats')
plt.xticks(rotation=45)
plt.show()
print("Mercedes, Tesla and Nissan have the highest number of seats and Smart the lowest")


# In[40]:


## Price of cars (in INR)

ax= plt.figure(figsize=(20,5))
sns.barplot(x='Brand',y='price_INR',data=dfev,palette='Set2')
plt.title('Price of a Car')
plt.xlabel('Car Brand')
plt.grid(axis='y')
plt.ylabel('Price in INR')
plt.xticks(rotation=45)
plt.show()
print("Lightyear, Porsche and Lucid are the most expensive and SEAT and Smart the least")


# In[41]:


dfev['PlugType'].value_counts()


# In[42]:


## Type of Plug used for charging

dfev['PlugType'].value_counts().plot.pie(figsize=(8,15),autopct='%.0f%%',explode=(.1,.1,.1,.1))
plt.title('Plug Type')
plt.show()
print('Most companies use Type 2 CCS and Type 1 CHAdeMo the least')


# In[43]:


dfev['BodyStyle'].value_counts()


# In[44]:


## Cars and their body style

dfev['BodyStyle'].value_counts().plot.pie(figsize=(8,15),autopct='%.0f%%',explode=(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1))
plt.title('Body Style')
plt.show()
print('Most cars are eiher SUV or Hatchback')


# In[45]:


dfev['Segment'].value_counts()


# In[46]:


## Segment in which the cars fall under

dfev['Segment'].value_counts().plot.pie(figsize=(8,15),autopct='%.0f%%',explode=(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1))
plt.title('Segment')
plt.show()
print('Most cars are either C or B type')


# In[47]:


df1 = dfev[["TopSpeed_KmH", "price_INR"]].groupby("TopSpeed_KmH").count()
df1


# In[48]:


df2= dfev[["Range_Km", "price_INR"]].groupby("Range_Km").count()
df2


# In[49]:


df3= dfev[["Range_Km", "TopSpeed_KmH"]].groupby("Range_Km").count()
df3


# In[50]:


df1 = df1.sort_values("TopSpeed_KmH",ascending = False).head(10)
df1


# In[51]:


df2=df2.sort_values("Range_Km",ascending = False).head(10)
df2


# In[52]:


df3=df3.sort_values("Range_Km",ascending = False).head(10)
df3


# In[53]:


plt.figure(figsize=(10,7))
plt.title('Cost based on top speed')
plt.pie(x=df1["price_INR"],labels=df1.index,autopct='%1.0f%%')
plt.show()


# In[54]:


plt.figure(figsize=(10,7))
plt.title('Cost based on Maximum Range')
plt.pie(x=df2["price_INR"],labels=df2.index,autopct='%1.0f%%')
plt.show()


# In[55]:


plt.figure(figsize=(10,7))
plt.title('Top Speeds based on Maximum Range')
plt.pie(x=df3["TopSpeed_KmH"],labels=df3.index,autopct='%1.0f%%')
plt.show()


# In[57]:


ax=plt.subplots(figsize=(15,8))
sns.stripplot(x='TopSpeed_KmH', y='FastCharge_KmH', data=dfev, jitter=True)


# In[58]:


ax=plt.subplots(figsize=(15,8))
sns.stripplot(x='TopSpeed_KmH', y='Efficiency_WhKm', data=dfev, jitter=True)


# In[59]:


dfev['PowerTrain'].replace(to_replace=['RWD','AWD','FWD'],value=[0, 2,1],inplace=True)


# In[60]:


dfev.head()


# In[61]:


features = ['AccelSec','TopSpeed_KmH','Efficiency_WhKm','FastCharge_KmH', 'RapidCharge','Range_Km', 'Seats', 'price_INR','PowerTrain']


# In[62]:


x = dfev.loc[:, features].values


# In[63]:


from sklearn.preprocessing import StandardScaler,PowerTransformer


# In[64]:


x = StandardScaler().fit_transform(x)


# In[65]:


x


# In[66]:


from sklearn.decomposition import PCA


# In[67]:


pca = PCA(n_components=9)
t = pca.fit_transform(x)
data2 = pd.DataFrame(t, columns=['PC1', 'PC2','PC3','PC4','Pc5','PC6', 'PC7', 'PC8','PC9'])
data2


# In[68]:


df_9=data2.iloc[:,:9]
df_9.head(3)


# In[70]:


# correlation coefficient between original variables and the component
loadings = pca.components_
num_pc = pca.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
loadings_dfev = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_dfev['variable'] = df_9.columns.values
loadings_dfev = loadings_dfev.set_index('variable')
loadings_dfev


# In[71]:


#Correlation matrix plot for loadings 
plt.rcParams['figure.figsize'] = (20,15)
ax = sns.heatmap(loadings_dfev, annot=True, cmap='Spectral')
plt.show()


# In[72]:


from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
from collections import Counter


# In[73]:


linked = linkage(data2, 'complete')
plt.figure(figsize=(13, 9))
dendrogram(linked, orientation='top')
plt.show()


# In[74]:


PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()


# In[75]:


import warnings
warnings.filterwarnings("ignore")


# In[76]:


model = KMeans(random_state=40)
visualizer = KElbowVisualizer(model, k=(2,9), metric='distortion', timings=True)
visualizer.fit(t)        # Fit the data to the visualizer
visualizer.show()  


# In[77]:


model = KMeans(random_state=40)
visualizer = KElbowVisualizer(model, k=(2,9), metric='silhouette', timings=True)
visualizer.fit(t)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


# In[78]:


model = KMeans(random_state=40)
visualizer = KElbowVisualizer(model, k=(2,9), metric='calinski_harabasz', timings=True)
visualizer.fit(t)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


# In[80]:


#K-means clustering 

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(t)
dfev['cluster_num'] = kmeans.labels_ #adding to df
print (kmeans.labels_) #Label assigned for each data point
print (kmeans.inertia_) #gives within-cluster sum of squares. 
print(kmeans.n_iter_) #number of iterations that k-means algorithm runs to get a minimum within-cluster sum of squares
print(kmeans.cluster_centers_) #Location of the centroids on each cluster.


# In[81]:


#To see each cluster size

Counter(kmeans.labels_)

#Visulazing clusters
sns.scatterplot(data=data2, x="PC1", y="PC9", hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.show()


# In[ ]:




