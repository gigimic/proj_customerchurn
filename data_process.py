import numpy as np
from numpy.core.fromnumeric import size 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Churn_Modelling.csv', delimiter = ',')
print(df.shape)
# print(df.isnull().sum())
print(df.nunique())
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1)
# print(df.head())


# making a pie chart showing the percentage of customers exited 
# labels = 'Exited', 'Retained'
# sizes = [df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
# explode = (0, 0.05)
# fig1, ax1 = plt.subplots(figsize=(10, 8))
# # ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
# #         shadow=True, startangle=90)
# ax1.axis('equal')
# plt.title("Proportion of customer churned and retained", size = 20)
# plt.show()

# Review the 'Status' relation with categorical variables
# fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
# sns.countplot(x='Geography', hue = 'Exited',data = df, ax=axarr[0][0])
# sns.countplot(x='Gender', hue = 'Exited',data = df, ax=axarr[0][1])
# sns.countplot(x='HasCrCard', hue = 'Exited',data = df, ax=axarr[1][0])
# sns.countplot(x='IsActiveMember', hue = 'Exited',data = df, ax=axarr[1][1])
# plt.show()
'''This plot shows people of which location are leaving the bank Memory
similarly gender, customers with card or active members
'''

# Relations based on the continuous data attributes
# fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
# sns.boxplot(y='CreditScore', x = 'Exited', hue = 'Exited', data = df, ax=axarr[0][0])
# sns.boxplot(y='Age', x = 'Exited', hue = 'Exited', data = df , ax=axarr[0][1])
# sns.boxplot(y='Tenure', x = 'Exited', hue = 'Exited', data = df, ax=axarr[1][0])
# sns.boxplot(y='Balance', x = 'Exited', hue = 'Exited', data = df, ax=axarr[1][1])
# sns.boxplot(y='NumOfProducts', x = 'Exited', hue = 'Exited', data = df, ax=axarr[2][0])
# sns.boxplot(y='EstimatedSalary', x = 'Exited', hue = 'Exited', data = df, ax=axarr[2][1])
# plt.show()

'''This plot shows customers with what credit score are leaving more. 
similary what age group, tenure, balance, no of products and salary
'''
