# Predition of customer churn
---------------------------

Here I try to identify the factors contribute to customer churn and build a machine learning model and predict if a customer is going to leave the bank or not.


Data:

10000 records and 14 features
Following are the features: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
4 of them are categorical variables: Geography, Gender, HasCrCard, IsActiveMember
5 of them are continuous variables: Age, Tenure, Balance, NumOfProducts, EstimatedSalary


## Exploratory data analysis:

Initially the percertage of the customers have churned is calculated.
Here 20% churned and 80% retained.

Then check the relation of categorical features with the 'Exited' number. Similary find the relation of continuous varibales.

The trends are shown in the following page:
https://github.com/gigimic/proj_customerchurn/blob/main/eda.pdf


## Feature engineering:

Try to find variables that can have an impact on the probability of churning.
For eg., i) ratio of balance by salary
ii) ratio of tenure by age
iii) ratio of credit score to age etc.


## Data preparation for model fitting:

One hot encode the categorical variables
Normalise the continuous variables (minmax scaling)

## Model fitting and selection:

Try a few models
Logistic regression 
SVM 
Ensemble models - Randomn forest, XG Boost classifier

## Hyperparameter tuning:

For each model, using sklearn gridsearchCV with arguments like model, parameter grid, and cross vadidation set. Then the best parameters are selected based on best_score.

The ROC (Receiver operating Characteristic) curve is then calculated and plotted. 
It is a plot between the true positive rates and false positive rates.
The optimum threshold can be decided.
The AUC for different models are then calculated and found the best model which gives the best prediction.

Then the selected model with the selected parameters can be used on the test data to make predictions.