# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 16:20:22 2025

@author: mkcar
"""

# Clean up
%reset -f
%clear

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


#Read data
df = pd.read_csv("LC_20.zip")

#Rid of unnecessary variables
keep_vars = [
    "loan_status",
    "annual_inc",
    "dti",
    "fico_range_low",
    "open_acc",
    "revol_util",
    "total_pymnt"
]

df= df[keep_vars]

#Drop missing values
df.dropna(inplace=True)

#Making sure data columns are gone
df.head()

#Split into train/test/validation
train_df, temp_df = train_test_split(df,test_size=0.4,random_state=123)
test_df,val_df = train_test_split(temp_df, test_size=0.5,random_state=123)


#Create variables for splits
X_train = train_df[["annual_inc", "dti", "fico_range_low", "open_acc", "revol_util", "total_pymnt"]]
y_train = train_df["loan_status"]

X_test  = test_df[["annual_inc", "dti", "fico_range_low", "open_acc", "revol_util", "total_pymnt"]]
y_test  = test_df["loan_status"]

X_val   = val_df[["annual_inc", "dti", "fico_range_low", "open_acc", "revol_util", "total_pymnt"]]
y_val   = val_df["loan_status"]

#Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)


#Logistic regression
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled,y_train)

#Get predict probabilities
prob = model.predict_proba(X_test_scaled)[:,1]

#Checking thresholds
thresholds = [0.65,0.7,0.75,0.8,0.85]

results = pd.DataFrame(columns=[
    "THRESHOLD", "accuracy", "true pos rate", "true neg rate", "false pos rate",
    "precision", "f-score"])

results["THRESHOLD"] = thresholds

j = 0

best_f = -1
best_thresh = None
best_cm = None

for i in thresholds:
    preds = np.where(model.predict_proba(X_test_scaled)[:,1] > i, 1, 0)

    cm = confusion_matrix(y_test, preds, labels=[1, 0])

    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]

    acc  = accuracy_score(y_test, preds)
    tpr  = recall_score(y_test, preds)           
    tnr  = TN / (TN + FP) if (TN + FP) > 0 else 0     
    fpr  = FP / (TN + FP) if (TN + FP) > 0 else 0       
    prec = precision_score(y_test, preds)
    f1   = f1_score(y_test, preds)
    
    # Store results
    results.iloc[j, 1] = acc
    results.iloc[j, 2] = tpr
    results.iloc[j, 3] = tnr
    results.iloc[j, 4] = fpr
    results.iloc[j, 5] = prec
    results.iloc[j, 6] = f1

    #Highest F-score
    if f1 > best_f:
        best_f = f1
        best_thresh = i
        best_cm = cm

    j += 1

#Print results
print(results.T)

#Print best confusion matrix
print("Best threshold:", best_thresh)
print("Best F-score:", best_f)
print("Confusion matrix at best threshold:")
print(best_cm)



