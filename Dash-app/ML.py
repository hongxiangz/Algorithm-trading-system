import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report


# Load Data Files to Dataframe
pd.set_option('display.max_rows', None)  # Show all rows
df1 = pd.read_csv("result.csv")
df2 = pd.read_csv("hw4_data.csv")
df1.rename(columns={"dt_enter":"Dates"}, inplace=True)
df1['Dates'] = pd.to_datetime(df1['Dates'])
df2['Dates'] = pd.to_datetime(df2['Dates'])
df1['Dates'] = df1['Dates'].dt.strftime('%Y/%-m/%-d')
df2['Dates'] = df2['Dates'].dt.strftime('%Y/%-m/%-d')
merged_df = pd.merge(df2, df1[["Dates", "success"]], on='Dates')
merged_df.set_index('Dates', inplace=True)

# Data Clean
merged_df.fillna(0, inplace=True)
merged_df.replace(-1, 0, inplace=True)
merged_df['SPXSFRCS Index'] = merged_df['SPXSFRCS Index'].astype(float)
# print(merged_df)

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(merged_df.iloc[:,0:9], merged_df['success'], test_size=0.2)

# Data Scaling
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# PCA 降维
transfer = PCA(n_components=0.99)
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# Train model
# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy-KNN:", accuracy)

# SVM
svm = SVC(kernel='linear', C=1, random_state=27)
svm.fit(x_train, y_train)
y_predict = svm.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy-SVM:", accuracy)

# LgR
lgr = LogisticRegression(random_state=11)
lgr.fit(x_train, y_train)
y_predict = lgr.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy-LgR:", accuracy)

# RF ***
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(x_train, y_train)
y_predict = rf.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy-RF:", accuracy)


