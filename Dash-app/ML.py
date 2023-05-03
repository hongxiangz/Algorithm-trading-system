import pandas as pd
from datetime import datetime

pd.set_option('display.max_rows', None)  # Show all rows\
df1 = pd.read_csv("/Users/hongxiangzhao/Desktop/result.csv")
df2 = pd.read_csv("/Users/hongxiangzhao/Desktop/hw4_data.csv")
df1.rename(columns={"dt_enter":"Dates"}, inplace=True)
df1['Dates'] = pd.to_datetime(df1['Dates'])
df1['Dates'] = df1['Dates'].dt.strftime('%Y/%-m/%-d')
merged_df = pd.merge(df2, df1[["Dates", "success"]], on='Dates')
print(merged_df)

