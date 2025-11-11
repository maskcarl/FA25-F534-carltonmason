# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:28:48 2025

@author: mkcar
"""

# Clean up
%reset -f
%clear

# Importing necessary packages
import pandas as pd # python's data handling package

#Read .csv and name
data = pd.read_csv('cardekho_trimmed.csv') 


#Drop the fields 'brand' and 'car_name'
df_new = data.drop(['brand','model'],axis=1)
#df_new.head()


#Convert the fields 'car_name', 'seller_type','fuel_type', and
#'transmission_type' to dummy variables
df_dummies1 = pd.get_dummies(df_new['car_name'], prefix='N')
df_dummies2 = pd.get_dummies(df_new['seller_type'], prefix='N')
df_dummies3 = pd.get_dummies(df_new['fuel_type'], prefix='N')
df_dummies4 = pd.get_dummies(df_new['transmission_type'], prefix='N')

#Concatenate dummy variables
df_encoded = pd.concat([df_new, df_dummies1,df_dummies2,df_dummies3,df_dummies4], axis=1)

#Drop categorical columns
df_encoded = df_encoded.drop(['car_name','seller_type','fuel_type','transmission_type'], axis=1)


#Break the dataframe into a 5000-record set for training and
#remaining for validation
train = df_encoded.iloc[:5000] 
val = df_encoded.iloc[5000:15277]


#Use linear regression to model 'selling_price' as function
#of the other fields on the training set
X_train, X_val = train.drop('selling_price', axis=1), val.drop('selling_price', axis=1)
y_train, y_val = train[['selling_price']], val[['selling_price']] 

# Importing models
from sklearn.linear_model import LinearRegression

lr_train=LinearRegression()
lr_train.fit(X_train,y_train)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1)


#Calculate the MSE of the model on training set and print out
y_pred = lr_train.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
print(mse)


#Apply the model to the observations in validation set
lr_val=LinearRegression()
lr_val.fit(X_val,y_val)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1)

#Calculate the MSEs of the model on validation set and print out
y_pred1=lr_val.predict(X_val)
mse = mean_squared_error(y_val,y_pred1)
print(mse)
