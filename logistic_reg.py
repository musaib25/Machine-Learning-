# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 07:08:32 2022

@author: Raisa
"""

import os
os.getcwd()
os.chdir("C:/Users/Raisa/Documents/jerry(DS)/datasets")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mydata=pd.read_csv("titanic.csv")

df = np.corrcoef()

print(mydata.dtypes)
corr_ht=mydata.corr()
print(corr_ht)
sns.heatmap(corr_ht,annot=True)
plt.show()

#feature selection-drop columns that are not required by checking correlation
finaldata = mydata.drop(["PassengerId","Pclass","Name","SibSp","Parch","Ticket","Fare","Cabin"],axis=1)
print(finaldata.isnull().sum())
plt.boxplot(finaldata["Age"])
print(finaldata.Age.median())
finaldata["Age"].fillna(finaldata.median(),inplace=True)
finaldata["Embarked"].fillna("S",inplace=True)

#frequency of each actegory of embarked
finaldata["Embarked"].value_counts()
sns.countplot(x='Embarked', data=finaldata)

y=finaldata[["Survived"]]
x=finaldata.drop(["Survived"],axis=1)
x=pd.get_dummies(x)
print(x.isnull().sum())
print(x.Age.median())
x["Age"] = x["Age"].fillna(0)
x["Age"].fillna(x.median(),inplace=True)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LogisticRegression
lm=LogisticRegression()
lm.fit(xtrain,ytrain)
predicted_value=lm.predict(xtest)

from sklearn.metrics import confusion_matrix
confusion_matrix(ytest,predicted_value)

from sklearn.metrics import classification_report
print(classification_report(ytest, predicted_value))
