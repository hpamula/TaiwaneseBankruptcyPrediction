# load data.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# load data

data = pd.read_csv('data.csv')
print(data.head())

# do matrix correlation to see which variables are correlated

corr = data.corr()
print(corr)

# draw a heatmap to see the correlation
sns.heatmap(corr)
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', linewidths=.2)
plt.show()


# First column = Y (bankrupt or not)
# Rest of column (2-95) = X (financial ratios)
# Do XGBoost Classification

# split data into X and Y
X = data.iloc[:, 1:95]
Y = data.iloc[:, 0]

# split data into train and test
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=12)
X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=12)


# import XGBoost
from xgboost import XGBClassifier

# create model
model = XGBClassifier()

# fit model
model.fit(X_train, Y_train)

# make predictions for test data
y_pred = model.predict(X_test)

# evaluate predictions
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))



