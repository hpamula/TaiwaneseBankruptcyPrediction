import pandas as pd
import seaborn as sns
from scipy.stats import stats, zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from IPython.display import Javascript
import numpy as np
from xgboost import XGBClassifier

plt.interactive(False)
plt.style.use('dark_background')


class Classification:
    def __init__(self):
        data = pd.read_csv('data.csv')
        # First column = Y (bankrupt or not)
        # Rest of column (2-95) = X (financial ratios)
        self.X = data.iloc[:, 1:95]
        self.y = data.iloc[:, 0]

    #      self.normalization()

    def correlation_matrix_and_plot(self, is_plot):
        """
        :return: Correlation matrix
        """

        corr = self.X.corr()

        if is_plot:
            print(corr)
            sns.heatmap(corr)
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr, cmap='coolwarm', linewidths=.2)
            plt.show()

        return corr

    def fillMissingValues(self):
        for column in self.X.isna().sum()[self.X.isna().sum() > 0].index.values:
            self.X[column].fillna(self.X[column].median(), inplace=True)
        print(self.X.isna().sum())

    def normalization(self):
        self.X = normalize(self.X.values, axis=0)
        self.y = normalize(self.y.values.reshape(-1, 1))

    def test_train_split(self):
        self.X_train, X_temp, self.Y_train, Y_temp = train_test_split(self.X, self.y, test_size=0.3, random_state=12)
        self.X_test, self.X_val, self.Y_test, self.Y_val = train_test_split(X_temp, Y_temp, test_size=0.5)

    def xgboost(self):
        model = XGBClassifier()
        model.fit(self.X_train, self.Y_train)

        # Make a prediction on validation set
        y_val_pred = model.predict(self.X_val)
        y_val_scores = model.predict_proba(self.X_val)[:, 1]

        # Plot the confusion matrix
        self.plot_confusion_matrix(model, self.Y_val, y_val_pred)

        # Plot the ROC curve
        self.plot_roc_curve(model, self.Y_val, y_val_scores)
        # evaluate predictions
        accuracy = accuracy_score(self.Y_val, y_val_pred)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

    def plot_roc_curve(self, model, y_true, y_score):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.0])
        plt.ylim([-0.1, 1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    def plot_confusion_matrix(self, model, y_true, y_pred):
        """

        :param model:
        :param y_true:
        :param y_pred:
        :return:
        """

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
     #   all_sample_title = 'Accuracy Score: {0}'.format(model.score(y_true, y_pred))
     #   plt.title(all_sample_title, size=15)
        plt.show()

    def remove_correlated_features(self, correlation_matrix):
        correlated_features = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > 0.8:
                    colname = correlation_matrix.columns[i]
                    if colname in self.X.columns:
                        print("Removing correlated feature %s" % colname)
                        correlated_features.add(colname)

        # Remove correlated features in X
        self.X = self.X.drop(columns=correlated_features)

    #  self.X_val = self.X_val.drop(columns=correlated_features)
    #  self.X_test = self.X_test.drop(columns=correlated_features)

    # def remove_outliers(self):
    #     for column in self.X.columns:
    #         if column != 'Bankrupt?':
    #             z = np.abs(zscore(self.X[column]))
    #             threshold = 3
    #             print(np.where(z > 3))
    #             print(self.X[column].iloc[np.where(z > 3)])
    #             self.X = self.X[(z < 3)]
    #             print(self.X)

    def feedForwardNN(self):
        pass


c = Classification()
# c.correlation()
# # Exploratory data analysis
correlation_matrix = c.correlation_matrix_and_plot(False)
c.remove_correlated_features(correlation_matrix)
correlation_matrix2 = c.correlation_matrix_and_plot(True)

c.test_train_split()
c.xgboost()
