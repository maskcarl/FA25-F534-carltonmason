# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Clean up
%reset -f
%clear

# Importing necessary packages
import pandas as pd # python's data handling package
import numpy as np # python's scientific computing package
import matplotlib.pyplot as plt # python's plotting package

# Importing models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Reading from a CSV File 
data = pd.read_csv('df_trip.csv') 
data.head()

#Split data into training and validation set
#Approximately 75% for train and 25% for validation
train = data.iloc[:1770] 
val = data.iloc[1770:]

#Creating x and y variables
X_train, X_val = train.drop('tip_amount', axis=1), val.drop('tip_amount', axis=1)
y_train, y_val = train[['tip_amount']], val[['tip_amount']] 

#Lasso regression, using lambdas between 0.01 and 0.2
alphas=[0.01/2, 0.02/2, 0.03/3, 0.04/2, 0.05/2, 0.06/2, 0.07/2, 0.08/2, 0.09/2, 0.1/2, 
        0.11/2, 0.12/2, 0.13/2, 0.14/2, 0.15/2, 0.16/2, 0.17/2, 0.18/2, 0.19/2, 0.2/2]

#Create list for MSEs
mses=[]

#Run the lasso regression and print for each alpha value
for alpha in alphas:
    lasso=Lasso(alpha=alpha)
    lasso.fit(X_train,y_train)
    pred=lasso.predict(X_val)
    mses.append(mean_squared_error(y_val,pred))
    print("MSE for alpha", alpha, ":", mean_squared_error(y_val, pred))

#Finding best alpha (smallest MSE)
best_index = np.argmin(mses)
best_alpha = alphas[best_index]
print("Best alpha based on validation MSE:", best_alpha)

#Lasso using best alpha
final_lasso = Lasso(alpha=best_alpha)
final_lasso.fit(X_train,y_train)

#Predict final MSE on validation set
final_pred = final_lasso.predict(X_val)
final_mse = mean_squared_error(y_val, final_pred)

#Average tip amount in validation set
avg_tip = y_val.mean().values[0]

#Comparison of MSE and average tip
print("Final model MSE on validation set:", final_mse)
print("Average tip amount in validation set:", avg_tip)
