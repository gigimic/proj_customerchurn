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

Hyparameter tuning:

For each model, using sklearn gridsearchCV with arguments like model, parameter grid, and cross vadidation set.
Then the best parameters are selected based on best_score.

The ROC (Receiver operating Characteristic) curve is then calculated and plotted. 
It is a plot between the true positive rates and false positive rates.
The optimum threshold can be decided.
The AUC for different models are then calculated and found the best model 
which gives the best prediction.

Then the selected model with the selected parameters can be used on the test data to make predictions.


https://www.kaggle.com/kmalit/bank-customer-churn-prediction