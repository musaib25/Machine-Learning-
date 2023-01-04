import os
os.getcwd()
os.chdir("C:/Users/musai/Downloads")
os.getcwd()
import pandas as pd
telecom_data=pd.read_csv("Telecom_Data.csv")
y=telecom_data[["churn"]]
x=telecom_data.drop(["churn"])
x=pd.get_dummies(x)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)
!pip install xgboost
from xgboost import XGBClassifier
model=XGBClassifier()
model.fit(xtrain,ytrain)
predicted_value=model.predict(xtest)
from sklearn.metrics import confusion_matrix
confusion_matrix(ytest,predicted_value)
from sklearn.metrics import classification_report
classification_report(ytest,predicted_value)

pip install tensorflow

pip install scipy
