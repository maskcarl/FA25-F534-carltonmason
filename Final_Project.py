# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 14:55:08 2025

@author: mkcar
"""

#Clean up
%reset -f
%clear

#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Read data
df = pd.read_csv("LC_20.zip")

#List variables to keep
keep_vars = [
    "loan_status",
    "annual_inc",
    "dti",
    "fico_range_low",
    "open_acc",
    "revol_util",
    "total_pymnt"
]

#Keep variables and drop others
df = df[keep_vars].copy()

# Drop missing values
df.dropna(inplace=True)

# Separate variables
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

#Split data into 60:20:20
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=123, stratify=y
)

X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=123, stratify=y_temp
)

#Decision tree creation
clf = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    min_samples_split=1000,
    min_samples_leaf=200,
    random_state=0
)

clf = clf.fit(X_train,y_train)

fig,ax = plt.subplots(figsize=(40,30))
plot_tree(clf,filled=True, feature_names=X_train.columns,
          proportion=True)
plt.savefig("LC_Decision_Tree.png",dpi=300)
plt.show()

print("Saved decision tree image")

#Testing thresholds
THRESHOLD = [0.60, 0.70, 0.80]

results_dt = pd.DataFrame(index=range(len(THRESHOLD)),
                          columns=["THRESHOLD","accuracy","true pos rate","true neg rate",
                                   "false pos rate","precision","f-score"])

results_dt["THRESHOLD"] = THRESHOLD

#Validation set for choosing best Z
n_val = len(y_val)
Q_val = clf.predict_proba(X_val)[:,1]

j=0
bestZ = None
bestF1=-1

for i in THRESHOLD:
    preds = np.where(Q_val>i,1,0)
    
    cm = (confusion_matrix(y_val,preds,labels=[1,0])/n_val)*100
    
    print("Confusion matrix for threshold =",i)
    print(cm)
    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]

    acc = accuracy_score(y_val, preds)
    rec = recall_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)

    results_dt.iloc[j, 1] = acc
    results_dt.iloc[j, 2] = rec
    results_dt.iloc[j, 3] = TN / (FP + TN) if (FP + TN) != 0 else np.nan
    results_dt.iloc[j, 4] = FP / (FP + TN) if (FP + TN) != 0 else np.nan
    results_dt.iloc[j, 5] = prec
    results_dt.iloc[j, 6] = f1

    if f1 > bestF1:
        bestF1 = f1
        bestZ = i

    j += 1

print(results_dt.T.to_string(header=False))

print("Best Z =", bestZ, "with F1 =", round(bestF1,4))


#KNN
#Scale data
scaler= StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
X_val_s = scaler.transform(X_val)

best_k = None
best_f1_knn = -1

#Run for K 1-9
for k in range(1, 9):
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train_s, y_train)

    val_pred = knn.predict(X_val_s)
    f1 = f1_score(y_val, val_pred)

    print("K =",k," | Validation F1 =",f1)

    if f1 > best_f1_knn:
        best_f1_knn = f1
        best_k = k

print("Best KNN: K =", best_k,"| Validation F1 =",best_f1_knn)

#Fit best KNN and test
best_knn = KNeighborsClassifier(n_neighbors=best_k,n_jobs=-1)
best_knn.fit(X_train_s,y_train)
test_pred = best_knn.predict(X_test_s)

print(confusion_matrix(y_test,test_pred))
print(classification_report(y_test,test_pred))

















