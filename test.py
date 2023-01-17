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

# Based on each city of brazil make a choropleth map that shows the number of appointments in each city it should be interactive
df_copy = df.copy()
# make a choropleth map
import plotly.express as px
fig = px.choropleth(df_copy, locations="Neighbourhood", locationmode="Brazil", color="Neighbourhood", hover_name="Neighbourhood", animation_frame="Neighbourhood")
fig.show()

# show the rows where the age is less than 0 and count them, just 
# to make sure that there are no negative values
df[df['Age'] < 0].count()

# what is the highest age in the dataset
df['Age'].max()

# get a median value for the age column starting from the age 76
df[df['Age'] > 75]['Age'].median()