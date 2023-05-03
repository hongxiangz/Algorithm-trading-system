import pandas as pd
import numpy as np
import datetime
import math
import time

def mcszyr(df):
    df = df.set_index('trade_id')
    result = pd.DataFrame(columns=('trade_id','asset','dt_enter','dt_exit','success','n','rtn'))
    for i in df.index.unique():
        trade_id = i
        if i==df.index.unique()[-1]:
            asset = df.loc[i]['asset']
            et_date = df.loc[i]['date']
            ex_date = np.nan
            success = np.nan
            n = np.nan
            rtn = np.nan
            temp = {'trade_id':trade_id,'asset':asset,'dt_enter':et_date,'dt_exit':ex_date,'success':success,'n':n, 'rtn':rtn}
            result = result.append(temp,ignore_index=True)
            break
        asset = df.loc[i].iloc[0]['asset']
        et_date = df.loc[i].iloc[0]['date']
        if df.loc[i].iloc[1]['status']=='CANCELLED':
            ex_date = np.nan
            success = 0
            n = len(pd.bdate_range(datetime.datetime.strptime(df.loc[i].iloc[0]['date'],'%m/%d/%y'),datetime.datetime.strptime(df.loc[i].iloc[1]['date'],'%m/%d/%y')))
            rtn = np.nan
        elif df.loc[i].iloc[-1]['status']=='LIVE':
            ex_date = np.nan
            success = np.nan
            n = np.nan
            rtn = np.nan
        elif df.loc[i].iloc[-1]['type']=='MKT':
            ex_date = df.loc[i].iloc[-1]['date']
            success = -1
            n = len(pd.bdate_range(datetime.datetime.strptime(df.loc[i].iloc[0]['date'],'%m/%d/%y'),datetime.datetime.strptime(df.loc[i].iloc[-1]['date'],'%m/%d/%y')))
            rtn = math.log(df.loc[i].iloc[-1]['price']/df.loc[i].iloc[0]['price'])/n
        else:
            ex_date = df.loc[i].iloc[-1]['date']
            success = 1
            n =len(pd.bdate_range(datetime.datetime.strptime(df.loc[i].iloc[0]['date'],'%m/%d/%y'),datetime.datetime.strptime(df.loc[i].iloc[-1]['date'],'%m/%d/%y')))
            rtn = math.log(df.loc[i].iloc[-1]['price']/df.loc[i].iloc[0]['price'])/n
        temp = {'trade_id':trade_id,'asset':asset,'dt_enter':et_date,'dt_exit':ex_date,'success':success,'n':n, 'rtn':round(100*rtn,2)}
        result = result.append(temp,ignore_index=True)
    return result


blotter = pd.read_csv('blotter.csv')
ledger = mcszyr(blotter)
print(ledger)