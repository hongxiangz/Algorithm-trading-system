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


def ml_return(ledger):

    # Load Data Files to Dataframe
    df1 = ledger.drop(columns=ledger.columns[0], axis=1) # should remove first col !!
    df2 = pd.read_csv("hw4_data_logreturn.csv", na_values=["#DIV/0!", "#NUM!"])
    df2 = df2.iloc[:, :-8]
    df1.rename(columns={"dt_enter": "Dates"}, inplace=True)
    df1['Dates'] = pd.to_datetime(df1['Dates'])
    df2['Dates'] = pd.to_datetime(df2['Dates'])
    df1['Dates'] = df1['Dates'].dt.strftime('%Y/%-m/%-d')
    df2['Dates'] = df2['Dates'].dt.strftime('%Y/%-m/%-d')
    merged_df = pd.merge(df2, df1[["Dates", "success"]], on='Dates')

    # Data Preprocessing
    merged_df.fillna(0, inplace=True)

    # Split Dataset
    split_idx_1 = int(0.6 * len(merged_df))  # 60% of the rows
    split_idx_2 = int(0.4 * len(merged_df))  # 40% of the rows
    df_train = merged_df.iloc[:split_idx_1]
    df_test = merged_df.iloc[split_idx_1:split_idx_1 + split_idx_2]
    df_dates = df_test.Dates
    y_train = df_train.success
    y_test = df_test.success
    x_train = df_train.drop(["success", "Dates"], axis=1)
    x_test = df_test.drop(["success", "Dates"], axis=1)

    # Data Scaling
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # PCA 降维
    transfer = PCA(n_components=0.99)
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # Import Model
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(x_train, y_train)
    y_predict = rf.predict(x_test)

    # Output Predict
    df_success = pd.DataFrame(y_predict, columns=['success'])
    df_success.replace(-1, 0, inplace=True)
    df_Dates = pd.DataFrame(df_dates, columns=['Dates'])
    df_Dates = df_Dates.reset_index()
    df_Dates = df_Dates.drop(df_Dates.columns[0], axis=1)
    df_prdict = df_Dates.join(df_success)

    df_rtn = df1[['Dates', 'rtn']]
    df_rtn = df_rtn.tail(int(len(df_rtn) * 0.4))
    df_rtn.fillna(0, inplace=True)
    df_rtn = df_rtn.reset_index()
    df_rtn = df_rtn.drop(columns='index', axis=1)
    df_rtn['rtn'] = df_rtn['rtn'].multiply(df_prdict['success'])
    df_predict_rtn = df_rtn

    return df_predict_rtn