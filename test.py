import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, preprocessing

df = pd.read_csv('dataset-noshow.csv')
df.head()

# Show the unique values of each column except the 1st 2 columns
for col in df.columns[2:]:
    print(col)
    print(df[col].unique())

# Show the number of missing values in each column
df.isnull().sum()

# show the rows where the age is less than 0 and count them, just 
# to make sure that there are no negative values
df[df['Age'] < 0].count()

# what is the highest age in the dataset
df['Age'].max()

# get a median value for the age column starting from the age 76
df[df['Age'] > 75]['Age'].median()

# make a correlation plot for df
corr = df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

# count the number of appointments based on the cities with less than a 1000 entries
df['Neighbourhood'].value_counts()[df['Neighbourhood'].value_counts() < 1000]

# replace the cities with less than 1000 entries with 'Other'
df['Neighbourhood'] = df['Neighbourhood'].apply(lambda x: 'Other' if x in df['Neighbourhood'].value_counts()[df['Neighbourhood'].value_counts() < 1000].index else x)

# create Columns for each Neighbourhood
df = pd.get_dummies(df, columns=['Neighbourhood'])

