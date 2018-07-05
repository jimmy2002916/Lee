#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 16:17:58 2018

@author: jimmyhomefolder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("data/student_data.csv")
df_pred = df

label = 'Sauna' #要預測的 label
staying = 'Staying'
features = list(set(df.columns) - {label, 
                 'Name',
                 'Address',
                 'Working',
                 'Age of Parent',
                 'Consuming Power',
                 'Beauty',
                 'Purchased',
                 'Massage',
                 'Resource',
                 'Birth Date',
                 'Register Date'
                 })

#%%
# Encoding categorical data (we have string in our data type)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X_0 = LabelEncoder()
df['County'] = X_0.fit_transform(df['County'])
df['District'] = X_0.fit_transform(df['District'])
df['Road'] = X_0.fit_transform(df['Road'])
df['Sex'] = X_0.fit_transform(df['Sex'])
#X['Consuming Power'] = X_0.fit_transform(X['Consuming Power'])
#X['Resource '] = X_0.fit_transform(X['Resource '])
#X['Sentimental'] = X_0.fit_transform(X['Sentimental'])

#%%
#df['Birth Date'] = pd.to_datetime(df['Birth Date'])
#df['Register Date'] = pd.to_datetime(df['Register Date'])
df = df.loc[df[staying].notnull()] # 我覺得staying 是個很強的特徵，先用它有值的資料來做預測 看看效果
df_staying_notnull = df
df = df.loc[df[label].notnull() ] # 要預測的 label
X = df[features] #metrics of features
y = df[label].values # independent variable vector(outcome) #Label Column

#%%
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%%
# Fitting Logistic Regression to the Training set

import keras
from keras.models import Sequential #used to initialize our neural network
from keras.layers import Dense # used to create layers in our neural network
from keras.layers import Dropout

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


"""
input_dim = len(X.columns)
classifier = Sequential() # classifier is the future ann we r going to build
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_dim))
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 1, epochs = 25)


classifier = GaussianNB()
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier = LogisticRegression(random_state = 0)
"""

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)

#%%
# Making the Confusion Matrix(it is a function)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred.round())

#%%
### Predicting...
df_pred = df_pred.loc[df_pred[staying].notnull()] # 我覺得staying 是個很強的特徵，先用它有值的資料來做預測 看看效果
X_pred = df_pred[features] #metrics of features
X_pred_df = X_pred

#%%
X_pred = X_pred.values #change pandas into numpy

Sauna_prediction = []
for i in X_pred[0:]:
    new_prediction = classifier.predict(sc.transform(np.array([list(i)])))
    new_prediction = (new_prediction > 0.5)
    Sauna_prediction.append(new_prediction)

#%%
df_pred['Sauna_prediction'] = Sauna_prediction
df_pred.to_excel('Sauna_prediction.xlsx')