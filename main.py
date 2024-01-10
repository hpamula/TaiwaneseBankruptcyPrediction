import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from IPython.display import Javascript
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
plt.interactive(False)
plt.style.use('dark_background')

class Classification:
    def __init__(self):
        self.data = pd.read_csv('data.csv')
        # First column = Y (bankrupt or not)
        # Rest of column (2-95) = X (financial ratios)
        self.X = self.data.iloc[:, 1:95]
        self.y = self.data.iloc[:, 0]
    def correlation(self):
        corr = self.data.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap='coolwarm', linewidths=.2)
        plt.xticks([])  # Remove x-axis labels
        plt.yticks([])  # Remove y-axis labels
        plt.show()
        return corr
    def fillMissingValues(self):
        for column in self.X.isna().sum()[self.X.isna().sum() > 0].index.values:
            self.X[column].fillna(self.X[column].median(), inplace=True)
        # print(self.X.isna().sum())
    def normalization(self):
        self.X = normalize(self.X.values, axis=0)
        self.y = normalize(self.y.values.reshape(-1, 1))
    def test_train_split(self):
        self.X_train, X_temp, self.Y_train, Y_temp = train_test_split(self.X, self.y, test_size=0.3, random_state=12)
        self.X_test, self.X_val, self.Y_test, self.Y_val = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=12)
    
    def xgboost(self):
        # Ratio of negative and positive samples in the training set
        ratio = len(self.Y_train[self.Y_train == 0]) / len(self.Y_train[self.Y_train == 1])
        # Ratio = 28.46 - czyli jest 28 razy wiecej firm niebankrutujacych niz bankrutujacych
        print("Ratio: ", ratio)

        model = XGBClassifier(objective='binary:logistic', seed=12,
                              scale_pos_weight=ratio,
                              reg_alpha=1e-4,
                              reg_lambda=1e-4,
                              gamma=0.1,
                              max_depth=3,
                              learning_rate=0.2,
                              subsample=0.8)
        model.fit(self.X_train, self.Y_train)

        self.investigate_feature_importance(model)

        # Make a prediction on validation set
        y_val_scores = model.predict_proba(self.X_val)[:, 1]

        # Zmniejszamy threshold, bo przeoczenie bankructwa jest bardziej ryzykowne
        y_val_pred = (y_val_scores > 0.35).astype(int)

        self.plot_confusion_matrix(self.Y_val, y_val_pred)
        self.plot_roc_curve(self.Y_val, y_val_scores)

        # evaluate predictions
        accuracy_train = accuracy_score(self.Y_train, model.predict(self.X_train))
        accuracy_validation = accuracy_score(self.Y_val, y_val_pred)
        print("Accuracy Train: %.2f%%" % (accuracy_train * 100.0))
        print("Accuracy Validation: %.2f%%" % (accuracy_validation * 100.0))

    def random_forest_classifier(self):
        model = RandomForestClassifier(n_estimators=110, max_depth=5, random_state=12, class_weight='balanced')
        model.fit(self.X_train, self.Y_train)

        y_val_scores = model.predict_proba(self.X_val)[:, 1]
        y_val_pred = (y_val_scores > 0.5).astype(int)

        #      self.plot_confusion_matrix(self.Y_val, y_val_pred)
        #      self.plot_roc_curve(self.Y_val, y_val_pred)

        accuracy_train = accuracy_score(self.Y_train, model.predict(self.X_train))
        accuracy_validation = accuracy_score(self.Y_val, y_val_pred)
        print("Accuracy Train: %.2f%%" % (accuracy_train * 100.0))
        print("Accuracy Validation: %.2f%%" % (accuracy_validation * 100.0))

        # Test the model
        print()

        y_test_scores = model.predict_proba(self.X_test)[:, 1]
        y_test_pred = (y_test_scores > 0.5).astype(int)

        self.plot_roc_curve(self.Y_test, y_test_pred)
        self.plot_confusion_matrix(self.Y_test, y_test_pred)

        accuracy_test = accuracy_score(self.Y_test, y_test_pred)
        print("Accuracy Test: %.2f%%" % (accuracy_test * 100.0))

    def investigate_feature_importance(self, model):
        feature_names = self.X.columns
        feature_importance = model.feature_importances_

        feature_importance_df = pd.DataFrame({'feature': feature_names,
                                              'importance': feature_importance})

        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis', hue='feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
        print(feature_importance_df)

        median = feature_importance_df['importance'].median()  # quantile(0.5)
        quantile_25 = feature_importance_df['importance'].quantile(0.25)
        print("Median: ", median)
        print("Quantile 25: ", quantile_25)

        # Remove features with importance score less than quantile 25
        self.X = self.X[feature_importance_df[feature_importance_df['importance'] > median]['feature'].values]
        print(self.X.columns)

        # draw the new feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature',
                    data=feature_importance_df[feature_importance_df['importance'] > median], palette='viridis',
                    hue='feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_roc_curve(y_true, y_score):
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

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        # Extract TN, FP, FN, TP
        tn, fp, fn, tp = cm.ravel()
        print("True Negative: ", tn)
        print("False Positive: ", fp)
        print("False Negative: ", fn)
        print("True Positive: ", tp)

        # Calculate sensitivity, specificity, and precision
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        print("Sensitivity: ", sensitivity)
        print("Specificity: ", specificity)
        print("Precision: ", precision)

        plt.show()

    def remove_correlated_features(self, correlation_matrix, threshold=0.8):
        correlated_features = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    colname = correlation_matrix.columns[i]
                    if colname in self.X.columns:
                        # print("Removing correlated feature %s" % colname)
                        correlated_features.add(colname)

        # Remove correlated features in X
        self.X = self.X.drop(columns=correlated_features)

    #  self.X_val = self.X_val.drop(columns=correlated_features)
    #  self.X_test = self.X_test.drop(columns=correlated_features)

    def feedForwardNN(self):
        pass
        
c = Classification()
c.fillMissingValues()
c.remove_correlated_features(c.correlation())
# c.normalization()
c.test_train_split()
c.random_forest_classifier()