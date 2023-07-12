# Penguin-Analysis
# Step 1: Import the Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Step 2: Load the Datasets
penguins_iter_df = pd.read_csv('penguins_lter.csv')
penguins_size_df = pd.read_csv('penguins_size.csv')

# Step 3: Explore the Datasets
# Display the first few rows of the DataFrames
penguins_iter_df.head()
penguins_size_df.head()

penguins_iter_df.info()
penguins_size_df.info()

penguins_iter_df.describe()
penguins_size_df.describe()

# Step 4: Handle Missing Data
penguins_size_df.isnull().sum()
penguins_size_df = penguins_size_df.dropna()
penguins_size_df.isnull().sum()

# Check for missing values in Penguins iter dataset
print(penguins_iter_df.isnull().sum())
penguins_iter_df = penguins_iter_df.dropna()
print(penguins_iter_df.isnull().sum())

# Step 5: Data Cleaning
# Assign unique column names to Penguins Iter dataset
new_column_names = ['Column1', 'Column2', 'Column3', 'Column4', 'Column5', 'Column6', 'Column7', 'Column8', 'Column9', 'Column10', 'Column11', 'Column12', 'Column13', 'Column14', 'Column15', 'Column16', 'Column17']
penguins_iter_df.columns = new_column_names

# Step 6: Exploratory Data Analysis
correlation = penguins_iter_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.show()

# Plot Different Features
plt.hist(penguins_iter_df['Column1'], bins=20)
plt.xlabel('Column1')
plt.ylabel('Count')
plt.show()

sns.countplot(x='Column2', data=penguins_iter_df)
plt.xlabel('Column2')
plt.ylabel('Count')
plt.show()

sns.scatterplot(x='Column1', y='Column3', hue='Column2', data=penguins_iter_df)
plt.xlabel('Column1')
plt.ylabel('Column3')
plt.show()

# Detect Outliers / Missing Values
sns.boxplot(x='Column2', y='Column4', data=penguins_iter_df)
plt.xlabel('Column2')
plt.ylabel('Column4')
plt.show()

# Find Correlation Between Variables
correlation = penguins_size_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.show()

# EDA ON PENGUINS SIZE
# Histograms
plt.hist(penguins_size_df['culmen_length_mm'], bins=20)
plt.xlabel('culmen_length_mm')
plt.ylabel('Count')
plt.show()

# Bar plots
sns.countplot(x='species', data=penguins_size_df)
plt.xlabel('species')
plt.ylabel('Count')
plt.show()

# Scatter plots
sns.scatterplot(x='culmen_length_mm', y='culmen_depth_mm', hue='species', data=penguins_size_df)
plt.xlabel('culmen_length_mm')
plt.ylabel('culmen_length_mm')
plt.show()

# Detect Outliers / Missing Values
sns.boxplot(x='species', y='body_mass_g', data=penguins_size_df)
plt.xlabel('species')
plt.ylabel('body_mass_g')
plt.show()
