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


# # making a pie chart showing the percentage of customers exited 
# labels = 'Exited', 'Retained'
# sizes = [df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
# explode = (0, 0.05)
# fig1, ax1 = plt.subplots(figsize=(10, 8))
# ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=90)
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

# Relations based on the continuous data attributes
# fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
# sns.boxplot(y='CreditScore', x = 'Exited', hue = 'Exited', data = df, ax=axarr[0][0])
# sns.boxplot(y='Age', x = 'Exited', hue = 'Exited', data = df , ax=axarr[0][1])
# sns.boxplot(y='Tenure', x = 'Exited', hue = 'Exited', data = df, ax=axarr[1][0])
# sns.boxplot(y='Balance', x = 'Exited', hue = 'Exited', data = df, ax=axarr[1][1])
# sns.boxplot(y='NumOfProducts', x = 'Exited', hue = 'Exited', data = df, ax=axarr[2][0])
# sns.boxplot(y='EstimatedSalary', x = 'Exited', hue = 'Exited', data = df, ax=axarr[2][1])
# plt.show()

# Split Train, test data
df_train = df.sample(frac=0.8,random_state=200)
df_test = df.drop(df_train.index)
print(len(df_train))
print(len(df_test))

# feature engineering
# df_train['BalanceSalaryRatio'] = df_train.Balance/df_train.EstimatedSalary
# sns.boxplot(y='BalanceSalaryRatio',x = 'Exited', hue = 'Exited',data = df_train)
# plt.ylim(-1, 5)
# plt.show()
# here we can see that customers of high balancesalary ratio leave more 

# Given that tenure is a 'function' of age, we introduce a variable aiming to standardize tenure over age:
# df_train['TenureByAge'] = df_train.Tenure/(df_train.Age)
# sns.boxplot(y='TenureByAge',x = 'Exited', hue = 'Exited',data = df_train)
# plt.ylim(-1, 1)
# plt.show()


df_train['CreditScoreGivenAge'] = df_train.CreditScore/(df_train.Age)
sns.boxplot(y='CreditScoreGivenAge',x = 'Exited', hue = 'Exited',data = df_train)
# plt.ylim(-1, 1)
plt.show()

