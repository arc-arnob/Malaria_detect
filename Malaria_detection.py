# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:15:21 2020

@author: Arnob
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import joblib

dataframe = pd.read_csv("csv\dataset_processed.csv")
print(dataframe.head())

#Splitting dataset into Training set and Test set

x = dataframe.drop(["Label"],axis=1)
y = dataframe["Label"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20, random_state = 4)

# Build a Model

model = RandomForestClassifier(n_estimators = 100, max_depth = 6)
model.fit(x_train,y_train)

joblib.dump(model,"rf_malaria_100_5")

# Make Predictions

prediction = model.predict(x_test)

print(metrics.classification_report(prediction,y_test))
 
md = joblib.load("rf_malaria_100_5")
# use md object to load the pre-trained data and predict the class of the image.