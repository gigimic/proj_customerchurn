Predition of customer churn
---------------------------

As the first step, identify the factors contribute to customer churn
Build a preduction model and classify if a customer is going to churn or not


Data:

10000 records and 14 features
Following are the features
CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
4 of them are categorical variables: Geography, Gender, HasCrCard, IsActiveMember
5 of them are continuous variables: Age, Tenure, Balance, NumOfProducts, EstimatedSalary


Exploratory data analysis:

Initially find out what percertage of the customers have churned.
Here 20% churned and 80% retained.

Then check the relation of categorical features with the 'Exited' number
Similary find the relation of continuous varibales.
Find the trends.


Feature engineering:

Try to find variables that can have an impact on the probability of churning.
For eg., i) ratio of balance by salary
ii) ratio of tenure by age
iii) ratio of credit score to age etc.


Data preparation for model fitting:

One hot encode the categorical variables
Normalise the continuous variables (minmax scaling)


Model fitting and selection:

Try a few models
Logistic regression in the primal space and with different kernels
SVM in the primal and with different Kernels
Ensemble models - Randomn forest, XG Boost classifier

Calculate the accuracy of each model
Plot the ROC curve


https://www.kaggle.com/kmalit/bank-customer-churn-prediction