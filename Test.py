# Importing necessary packages and functions required
import numpy as np # for numerical computations
import pandas as pd # for data processing,I/O file operations
import matplotlib.pyplot as plt # for visualization of different kinds of plots
get_ipython().run_line_magic('matplotlib', 'inline')
# for matplotlib graphs to be included in the notebook, next to the code
import seaborn as sns # for visualization 
import warnings # to silence warnings
warnings.filterwarnings('ignore')


# In[2]:


# Importing red wine data into a dataframe
data=pd.read_csv("C:\Users\user\Documents\Python\HW7\HW_7\wine.data")


# In[3]:


# Glimpse of the data
data.sample(5)


# In[4]:


#shape of the data i.e., no of rows and columns in the data
data.shape


# In[5]:


#size of the data
data.size


# ### Data Analysis and Visualization

# In[6]:


# data information i.e., datatypes of different columns,their count etc
data.info()


# In[7]:


# Description of the data i.e., Descriptive Statistics
data.describe()


# In[8]:


# checking the different classes of the wine quality 
data.quality.unique()


# We observe there are a total of 6 unique wine qualities in our data.

# In[9]:


# Checking the number of supporting observations for each class of wine quality
data['quality'].value_counts()


# The wine quality 5 has the maximum supporting cases in the data of 681 cases,while the wine qualities 3,8 have very less number of supporting cases of 10,18 respectively. 

# In[10]:


sns.countplot(data.quality)
plt.show()


# From the above count plot we find that wines with normal quality(4,5,6,7) have more no of instances while the excellent or poor quality wines(8,3) respectively have less instances for support.

# Since the classes are not balanced,we remove the classes with less supporting classes i.e, qualities 3, 8 as they hinder the learning process of the models while fitting the data which produces abnormal results.

# In[11]:


# Get names of indexes for which column Age has value 30
indexNames1 = data[ (data['quality'] == 3) ].index
indexNames2 = data[ (data['quality'] == 8) ].index
 
# Delete these row indexes from dataFrame
data.drop(indexNames1, inplace=True)
data.drop(indexNames2,inplace=True)


# In[12]:


# Checking the number of supporting observations for each class of wine quality
data['quality'].value_counts()


# In[13]:


sns.countplot(data.quality)
plt.show()


# Now each class have a decent number of supporting classes for the model to learn and classify a new one.

# In[14]:


# Now lets see the shape of the data
data.shape


# In[15]:


# Checking for missing values in the data
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap="viridis")
plt.show()


# From the above heat map we observe that there are no missing values in the data.

# In[16]:


# Checking fixed acidity levels for each wine quality
fig = plt.figure(figsize = (8,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = data)


# In[17]:


# Takes more run time can avoid this code if considered unnecessary.
fig,ax=plt.subplots(4,2,figsize=(15,15))
plt.subplots_adjust(hspace=.4)
ax[0,0].bar(x='quality',height='fixed acidity',data = data)
ax[0,1].bar(x="quality",height="volatile acidity",data=data)
ax[1,0].bar(x="quality",height="citric acid",data=data)
ax[1,1].bar(x="quality",height="residual sugar",data=data)
ax[2,0].bar(x="quality",height="chlorides",data=data)
ax[2,1].bar(x="quality",height="free sulfur dioxide",data=data)
ax[3,0].bar(x="quality",height="sulphates",data=data)
ax[3,1].bar(x="quality",height="alcohol",data=data)
ax[0,0].set_title("fixed acidity")
ax[0,1].set_title("volatile acidity")
ax[1,0].set_title("citric acid")
ax[1,1].set_title("residual sugar")
ax[2,0].set_title("chlorides")
ax[2,1].set_title("free sulfur dioxide")
ax[3,0].set_title("sulphates")
ax[3,1].set_title("alcohol")
plt.show()


# We can see various levels of different features(fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,sulphates,alcohol) for different kinds of wine quality.

# In[18]:


fig = plt.figure(figsize = (9,6))
sns.pointplot(x=data['pH'].round(1),y='residual sugar',color='green',data=data)
plt.show()


# From the above point plot we can see various point estimates and confidence levels for residual sugar levels at different values of pH.

# In[19]:


fig = plt.figure(figsize = (8,6))
sns.pointplot(y=data['pH'].round(1),x='quality',color='MAGENTA',data=data)
plt.show()


# At different wine qualities the point estimates and confidence intervals for pH values are shown in the above point plot.

# In[20]:


# Takes more run time can avoid this code if considered unnecessary. 
sns.pairplot(data)
plt.show()


# In[21]:


corr=data.corr()


# In[22]:


corr


# In[23]:


# Visualizing correlation
plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True)
plt.show()