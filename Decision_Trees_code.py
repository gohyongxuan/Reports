# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""Baseball Data from Kaggle"""

df = pd.read_csv('baseball.csv')

df.fillna(0, inplace=True)
df['OPS'] = df['OBP'] + df['SLG']
df['OOPS'] = df['OOBP'] + df['OSLG']
df['Year'] = df['Year'].apply(lambda x: int(x))
df.drop(['League','Team'],axis=1,inplace=True)
df.head()

# df1 = df.drop(['Year', 'RS', 'RA', 'W', 'OBP', 'SLG', 'RankSeason', 'RankPlayoffs', 'G'], axis=1)
df1 = df.drop(['Year', 'RS', 'RA', 'W', 'RankSeason', 'RankPlayoffs', 'G', 'OOPS', 'OPS'], axis=1)
df2 = df.drop(['Year', 'RS', 'RA', 'W', 'RankSeason', 'RankPlayoffs', 'G', 'OBP', 'SLG', 'OOBP', 'OSLG'], axis=1)
df2.head()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='OOPS', y='OPS', hue='Playoffs', palette={0: 'blue', 1: 'red'}, s=100, edgecolor='black')

X = df1.drop(columns=['Playoffs'])
y = df1['Playoffs']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_2 = df2.drop(columns=['Playoffs'])
y_2 = df2['Playoffs']
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42)

"""
Decision Tree Classifier with sklearn"""

dt_model = DecisionTreeClassifier(max_depth=3)
dt_model.fit(X_train, y_train)

y_pred1 = dt_model.predict(X_test)

print("Classification Report for Decision Tree Classifier:")
print(classification_report(y_test, y_pred1))

accuracy1 = accuracy_score(y_test, y_pred1)
print(f"Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(14, 10))
tree.plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=[str(label) for label in y.unique()], rounded=True)
plt.show()

"""Manual Decision Tree Classifier"""

def calculate_entropy(y):
    classes = y.value_counts()
    probabilities = classes / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def information_gain(X, y, feature, value):
    left_y = y[X[feature] <= value]
    right_y = y[X[feature] > value]
    left_entropy = calculate_entropy(left_y)
    right_entropy = calculate_entropy(right_y)
    weighted_entropy = (len(left_y) / len(y)) * left_entropy + (len(right_y) / len(y)) * right_entropy
    gain = calculate_entropy(y) - weighted_entropy
    return gain

def build_tree(X, y, max_depth=3, min_samples_split=2, current_depth=0):
    if current_depth == max_depth or len(X) < min_samples_split or len(y.unique()) == 1:
        return {'leaf': True, 'prediction': y.mode()[0]}
    best_feature = None
    best_value = None
    best_gain = -1
    for feature in X.columns:
        possible_values = X[feature].unique()
        for value in possible_values:
            gain = information_gain(X, y, feature, value)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_value = value
    if best_gain <= 0:
        return {'leaf': True, 'prediction': y.mode()[0]}
    left_mask = X[best_feature] <= best_value
    right_mask = ~left_mask
    left_X, right_X = X[left_mask], X[right_mask]
    left_y, right_y = y[left_mask], y[right_mask]
    left_tree = build_tree(left_X, left_y, max_depth, min_samples_split, current_depth + 1)
    right_tree = build_tree(right_X, right_y, max_depth, min_samples_split, current_depth + 1)
    return {'leaf': False, 'feature': best_feature, 'value': best_value, 'left': left_tree, 'right': right_tree}

def predict_tree(tree, X):
    if tree['leaf']:
        return tree['prediction']
    feature = tree['feature']
    value = tree['value']
    if X[feature] <= value:
        return predict_tree(tree['left'], X)
    else:
        return predict_tree(tree['right'], X)

def print_tree(tree, depth=0):
    if tree['leaf']:
        print(f"{'  ' * depth}Prediction: {tree['prediction']}")
    else:
        print(f"{'  ' * depth}Feature: {tree['feature']} <= {tree['value']}")
        print(f"{'  ' * depth}Left:")
        print_tree(tree['left'], depth + 1)
        print(f"{'  ' * depth}Right:")
        print_tree(tree['right'], depth + 1)

tree_model = build_tree(X_train, y_train, max_depth=3, min_samples_split=2)

y_pred2 = X_test.apply(lambda row: predict_tree(tree_model, row), axis=1)

accuracy2 = (y_pred2 == y_test).mean()
print(f'Accuracy: {accuracy * 100:.2f}%')

print_tree(tree_model)

"""Random Forest Classifier"""

rf_model = RandomForestClassifier(n_estimators=1000, max_depth=3)
rf_model.fit(X_train, y_train)

y_pred3 = rf_model.predict(X_test)

print("Classification Report for Random Forest Classifier:")
print(classification_report(y_test, y_pred3))

accuracy3 = accuracy_score(y_test, y_pred3)
print(f"Accuracy: {accuracy * 100:.2f}%")

rf_model = RandomForestClassifier(n_estimators=1000, max_depth=3)
rf_model.fit(X_train_2, y_train_2)

y_pred4 = rf_model.predict(X_test_2)

print("Classification Report for Random Forest Classifier:")
print(classification_report(y_test_2, y_pred4))

accuracy4 = accuracy_score(y_test_2, y_pred4)
print(f"Accuracy: {accuracy * 100:.2f}%")
