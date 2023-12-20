# !pip install ucimlrepo
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import roc_curve, auc
from IPython.display import Javascript
plt.interactive(False)
plt.style.use('dark_background')



# load data.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# load data
def Daniela():
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
def Huberta():
    print(heart_diseases.data.headers)
    # print(heart_diseases.data.original)
    X = heart_diseases.data.features # big letter for matrix
    print(X)
    y = heart_diseases.data.targets # small letter for vector
    print(y)

Huberta()