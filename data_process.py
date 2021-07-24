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
# Split Train, test data
df_train = df.sample(frac=0.8,random_state=200)
df_test = df.drop(df_train.index)
print(len(df_train))
print(len(df_test))

# feature engineering
df_train['BalanceSalaryRatio'] = df_train.Balance/df_train.EstimatedSalary
# sns.boxplot(y='BalanceSalaryRatio',x = 'Exited', hue = 'Exited',data = df_train)
# plt.ylim(-1, 5)
# plt.show()
# here we can see that customers of high balancesalary ratio leave more 

# Given that tenure is a 'function' of age, we introduce a variable aiming to standardize tenure over age:
df_train['TenureByAge'] = df_train.Tenure/(df_train.Age)
# sns.boxplot(y='TenureByAge',x = 'Exited', hue = 'Exited',data = df_train)
# plt.ylim(-1, 1)
# plt.show()

'''No calculate the ratio of  credit score given age to take into account credit behaviour 
of customers with age'''
df_train['CreditScoreGivenAge'] = df_train.CreditScore/(df_train.Age)
# sns.boxplot(y='CreditScoreGivenAge',x = 'Exited', hue = 'Exited',data = df_train)
# # plt.ylim(-1, 1)
# plt.show()
# print(df_train.head())

# data preparation for model fitting
# Arrange columns by data type for easier manipulation
continuous_vars = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio',
                   'TenureByAge','CreditScoreGivenAge']
cat_vars = ['HasCrCard', 'IsActiveMember','Geography', 'Gender']
df_train = df_train[['Exited'] + continuous_vars + cat_vars]
# print(df_train.head())

'''For the one hot variables, change 0 to -1 so that the models can capture a negative relation 
where the attribute in inapplicable instead of 0'''
df_train.loc[df_train.HasCrCard == 0, 'HasCrCard'] = -1
df_train.loc[df_train.IsActiveMember == 0, 'IsActiveMember'] = -1
# print(df_train.head())

# One hot encode the categorical variables
lst = ['Geography', 'Gender']
remove = list()
for i in lst:
    if (df_train[i].dtype == np.str or df_train[i].dtype == np.object):
        for j in df_train[i].unique():
            df_train[i+'_'+j] = np.where(df_train[i] == j,1,-1)
        remove.append(i)
df_train = df_train.drop(remove, axis=1)
# print(df_train.head())

# minMax scaling the continuous variables
minVec = df_train[continuous_vars].min().copy()
maxVec = df_train[continuous_vars].max().copy()
df_train[continuous_vars] = (df_train[continuous_vars]-minVec)/(maxVec-minVec)
print(df_train.head())

# data prep pipeline for test data
def DfPrepPipeline(df_predict,df_train_Cols,minVec,maxVec):
    # Add new features
    df_predict['BalanceSalaryRatio'] = df_predict.Balance/df_predict.EstimatedSalary
    df_predict['TenureByAge'] = df_predict.Tenure/(df_predict.Age - 18)
    df_predict['CreditScoreGivenAge'] = df_predict.CreditScore/(df_predict.Age - 18)
    # Reorder the columns
    continuous_vars = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary','BalanceSalaryRatio',
                   'TenureByAge','CreditScoreGivenAge']
    cat_vars = ['HasCrCard','IsActiveMember',"Geography", "Gender"] 
    df_predict = df_predict[['Exited'] + continuous_vars + cat_vars]
    # Change the 0 in categorical variables to -1
    df_predict.loc[df_predict.HasCrCard == 0, 'HasCrCard'] = -1
    df_predict.loc[df_predict.IsActiveMember == 0, 'IsActiveMember'] = -1
    # One hot encode the categorical variables
    lst = ["Geography", "Gender"]
    remove = list()
    for i in lst:
        for j in df_predict[i].unique():
            df_predict[i+'_'+j] = np.where(df_predict[i] == j,1,-1)
        remove.append(i)
    df_predict = df_predict.drop(remove, axis=1)
    # Ensure that all one hot encoded variables that appear in the train data appear in the subsequent data
    L = list(set(df_train_Cols) - set(df_predict.columns))
    for l in L:
        df_predict[str(l)] = -1        
    # MinMax scaling coontinuous variables based on min and max from the train data
    df_predict[continuous_vars] = (df_predict[continuous_vars]-minVec)/(maxVec-minVec)
    # Ensure that The variables are ordered in the same way as was ordered in the train set
    df_predict = df_predict[df_train_Cols]
    return df_predict

# model fitting

# Support functions
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform

# Fit models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Scoring functions
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Function to give best model score and parameters
def best_model(model):
    print(model.best_score_)    
    print(model.best_params_)
    print(model.best_estimator_)
def get_auc_scores(y_actual, method,method2):
    auc_score = roc_auc_score(y_actual, method); 
    fpr_df, tpr_df, _ = roc_curve(y_actual, method2); 
    return (auc_score, fpr_df, tpr_df)


# Fit logistic regression with degree 2 polynomial kernel
# param_grid = {'C': [0.1,10,50], 'max_iter': [300,500], 'fit_intercept':[True],'intercept_scaling':[1],'penalty':['l2'],
#               'tol':[0.0001,0.000001]}
# poly2 = PolynomialFeatures(degree=2)
# df_train_pol2 = poly2.fit_transform(df_train.loc[:, df_train.columns != 'Exited'])
# log_pol2_Grid = GridSearchCV(LogisticRegression(solver = 'liblinear'),param_grid, cv=5, refit=True, verbose=0)
# log_pol2_Grid.fit(df_train_pol2,df_train.Exited)
# best_model(log_pol2_Grid)

# Fit SVM with RBF Kernel
# param_grid = {'C': [0.5,100,150], 'gamma': [0.1,0.01,0.001],'probability':[True],'kernel': ['rbf']}
# SVM_grid = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=0)
# SVM_grid.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
# best_model(SVM_grid)

# Fit SVM with pol kernel
# param_grid = {'C': [0.5,1,10,50,100], 'gamma': [0.1,0.01,0.001],'probability':[True],'kernel': ['poly'],'degree':[2,3] }
# SVM_grid = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=0)
# SVM_grid.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
# best_model(SVM_grid)

# Fit random forest classifier
param_grid = {'max_depth': [3, 5, 6, 7, 8], 'max_features': [2,4,6,7,8,9],'n_estimators':[50,100],'min_samples_split': [3, 5, 6, 7]}
RanFor_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, refit=True, verbose=0)
RanFor_grid.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
best_model(RanFor_grid)