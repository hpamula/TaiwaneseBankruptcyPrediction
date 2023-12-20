import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import roc_curve, auc, accuracy_score
from IPython.display import Javascript
import numpy as np
from xgboost import XGBClassifier
plt.interactive(False)
plt.style.use('dark_background')

class Classification:
    def __init__(self):
        self.data = pd.read_csv('data.csv')
        # First column = Y (bankrupt or not)
        # Rest of column (2-95) = X (financial ratios)
        self.X = self.data.iloc[:, 1:95]
        self.y = self.data.iloc[:, 0]
        self.normalization()
    def correlation(self):
        print(self.data.head())
        corr = self.data.corr()
        print(corr)
        sns.heatmap(corr)
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap='coolwarm', linewidths=.2)
        plt.show()
    def xgboost(self):
        model = XGBClassifier()
        model.fit(self.X_train, self.Y_train)
        y_pred = model.predict(self.X_test)
        # evaluate predictions
        accuracy = accuracy_score(self.Y_test, y_pred)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
    def fillMissingValues(self):
        for column in self.X.isna().sum()[self.X.isna().sum() > 0].index.values:
            self.X[column].fillna(self.X[column].median(), inplace=True)
        print(self.X.isna().sum())
    def normalization(self):
        self.X = normalize(self.X.values, axis=0)
        self.y = normalize(self.y.values)
    def splitDataSet():
        X_train, X_temp, Y_train, Y_temp = train_test_split(X, y, test_size=0.3, random_state=12)
        X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=12)
    def feedForwardNN():
        
c = Classification()
c.correlation()