#!/usr/bin/env python
# coding: utf-8

# # Step 1: Import the Required Libraries
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# # Step 2: Load the Dataset's
# 

# In[2]:


penguins_iter_df = pd.read_csv('penguins_lter.csv')
penguins_size_df = pd.read_csv('penguins_size.csv')


# # Step 3: Explore the Datasets
# 

# In[3]:


# Display the first few rows of the DataFrames
penguins_iter_df.head()


# In[4]:


penguins_size_df.head()


# In[5]:


penguins_iter_df.info()


# In[6]:


penguins_size_df.info()


# In[7]:


penguins_iter_df.describe()


# In[8]:


penguins_size_df.describe()


# # Step 4: Handle Missing Data
# 

# In[9]:


penguins_size_df.isnull().sum()


# In[10]:


penguins_size_df = penguins_size_df.dropna()


# In[11]:


penguins_size_df.isnull().sum()


# In[12]:


# Check for missing values in Penguins iter dataset
print(penguins_iter_df.isnull().sum())


# In[13]:


penguins_iter_df = penguins_iter_df.dropna()


# In[14]:


print(penguins_iter_df.isnull().sum())


# # Step 5: Data Cleaning
# 

# In[15]:


# Assign unique column names to Penguins Iter dataset
new_column_names = ['Column1', 'Column2', 'Column3', 'Column4', 'Column5', 'Column6', 'Column7', 'Column8', 'Column9', 'Column10', 'Column11', 'Column12', 'Column13', 'Column14', 'Column15', 'Column16', 'Column17']
penguins_iter_df.columns = new_column_names


# # Step 6: Exploratory Data Analysis
# 

# In[16]:


correlation = penguins_iter_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.show()


# In[17]:


# Plot Different Features
plt.hist(penguins_iter_df['Column1'], bins=20)
plt.xlabel('Column1')
plt.ylabel('Count')
plt.show()


# In[18]:


sns.countplot(x='Column2', data=penguins_iter_df)
plt.xlabel('Column2')
plt.ylabel('Count')
plt.show()


# In[19]:


sns.scatterplot(x='Column1', y='Column3', hue='Column2', data=penguins_iter_df)
plt.xlabel('Column1')
plt.ylabel('Column3')
plt.show()


# ## Detect Outliers / Missing Values
# 

# In[20]:


# Detect Outliers / Missing Values
sns.boxplot(x='Column2', y='Column4', data=penguins_iter_df)
plt.xlabel('Column2')
plt.ylabel('Column4')
plt.show()


# ### Find Correlation Between Variables
# 

# In[21]:


correlation = penguins_size_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.show()


# ### EDA ON PENGUINS SIZE

# In[22]:


# Histograms
plt.hist(penguins_size_df['culmen_length_mm'], bins=20)
plt.xlabel('culmen_length_mm')
plt.ylabel('Count')
plt.show()


# In[23]:


# Bar plots
sns.countplot(x='species', data=penguins_size_df)
plt.xlabel('species')
plt.ylabel('Count')
plt.show()


# In[24]:


# Scatter plots
sns.scatterplot(x='culmen_length_mm', y='culmen_depth_mm', hue='species', data=penguins_size_df)
plt.xlabel('culmen_length_mm')
plt.ylabel('culmen_length_mm')
plt.show()


# ## Detect Outliers / Missing Values
# 

# In[25]:


# Detect outliers using box plots
sns.boxplot(x='species', y='body_mass_g', data=penguins_size_df)
plt.xlabel('species')
plt.ylabel('body_mass_g')
plt.show()

