# load data.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
