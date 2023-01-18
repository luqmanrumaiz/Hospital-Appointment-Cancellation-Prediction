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





from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(3) # Define classifier
knn.fit(train_features, train_labels) # Train model

# Make predictions
train_pred = knn.predict(train_features)
test_pred = knn.predict(test_features)


# get the accuracy score
print("Train Accuracy: ", accuracy_score(train_labels, train_pred))
print("Test Accuracy: ", accuracy_score(test_labels, test_pred))






















import plotly.express as px




for col in df.columns:
    if col != 'Cancelled':
        fig = px.histogram(df, x=col, color='Cancelled')
        fig.show()
# some chart that shows the ratio of noshow with each column
sns.countplot(x='No-show', data=df, palette='hls')
plt.show()


# get all the column maes from the dataset and store it in a list
cols = df.columns.tolist()

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

# make a stacked bar chart for each column
for col in df.columns:
    if col != 'Cancelled':
        df.groupby(col)['Cancelled'].mean().plot(kind='bar')
        plt.title(col)
        plt.show()


# count the number of appointments based on the cities with less than a 1000 entries
df['Neighbourhood'].value_counts()[df['Neighbourhood'].value_counts() < 1000]

# replace the cities with less than 1000 entries with 'Other'
df['Neighbourhood'] = df['Neighbourhood'].apply(lambda x: 'Other' if x in df['Neighbourhood'].value_counts()[df['Neighbourhood'].value_counts() < 1000].index else x)

# create Columns for each Neighbourhood
df = pd.get_dummies(df, columns=['Neighbourhood'])


# get unique count for SMS


# Make correlation plot for df pearson correlation
corr = df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

# create a new column for the day of the appointment for appointemtn date
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.dayofweek

# get unique  for day of the week
df['AppointmentDay'].unique()

# sort the df alphabetically by the column name
df = df.sort_values(by=['PatientId'])

# get unique values for the column 'Year'
df['Year'].unique()


# get the column names of test_features
# Displaying the Correlation Matrix in the form of a Heatmap, after removing Correlated Features to ensure that
# Correlated Features are removed

correlationMatrix = train_features.corr()
f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(correlationMatrix, annot=True, cmap=plt.cm.Reds)
plt.show()

# drop the column age
train_features = train_features.drop(['Age'], axis=1)


