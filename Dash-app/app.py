from dash import Dash, html, dcc, dash_table, Input, Output, State
import refinitiv.dataplatform.eikon as ek
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import plotly.express as px
import os
import refinitiv.dataplatform as rd

ek.set_app_key(os.getenv('EIKON_API'))

#dt_prc_div_splt = pd.read_csv('unadjusted_price_history.csv')

app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.A('Benchmark: '),
        dcc.Input(id = 'benchmark-id', type = 'text', value="IVV"),
        html.A('Asset: '),
        dcc.Input(id = 'asset-id', type = 'text', value="AAPL.O"),
        html.A('Date range: '),
        dcc.DatePickerRange(
            id='date-range',
            min_date_allowed=date(2015,1,1),
            max_date_allowed=datetime.today(),
            initial_visible_month=date(2023,1,1),
            start_date=date(2017,1,1),
            end_date=datetime.today()
        ),
        html.Div(id="output-container-date-picker-range")
    ]),
    html.Div([
        html.A('Date range for plot: '),
        dcc.DatePickerRange(
            id='new-date-range',
            min_date_allowed=date(2015,1,1),
            max_date_allowed=datetime.today(),
            initial_visible_month=date(2023, 1, 1),
            start_date=date(2017, 1, 1),
            end_date=datetime.today()
        ),
        html.Div(id="output-container-date-picker-range1")
    ]),
    
    html.Button('QUERY Refinitiv', id = 'run-query', n_clicks = 0),
    html.H2('Raw Data from Refinitiv'),
    dash_table.DataTable(
        id = "history-tbl",
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),
    html.H2('Historical Returns'),
    dash_table.DataTable(
        id = "returns-tbl",
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),

    html.H1('A-B plot'),
    html.H2('Alpha & Beta Scatter Plot'),
    dcc.Graph(id="ab-plot"),
    html.P(id='summary-text', children=""),
    html.H2('Alpha & Beta Value'),
    html.Div(id="alpha-beta-output"),

    html.H2('Blotters'),
    html.H6('ENTRY Blotters'),
    dash_table.DataTable(
        id="entry-blotter",
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),
    html.H6('EXIT Blotters'),
    dash_table.DataTable(
        id="exit-blotter",
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),

])


@app.callback(
    Output("history-tbl", "data"),
    Input("run-query", "n_clicks"),
    [State('benchmark-id', 'value'), State('asset-id', 'value'), State('date-range', 'start_date'), State('date-range', 'end_date')],
    prevent_initial_call=True
)
def query_refinitiv(n_clicks, benchmark_id, asset_id, start_date, end_date):
    assets = [benchmark_id, asset_id, start_date, end_date]
    prices, prc_err = ek.get_data(
        instruments=assets,
        fields=[
            'TR.OPENPRICE(Adjusted=0)',
            'TR.HIGHPRICE(Adjusted=0)',
            'TR.LOWPRICE(Adjusted=0)',
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters={
            'SDate': start_date,
            'EDate': end_date,
            'Frq': 'D'
        }
    )

    divs, div_err = ek.get_data(
        instruments=assets,
        fields=[
            'TR.DivExDate',
            'TR.DivUnadjustedGross',
            'TR.DivType',
            'TR.DivPaymentType'
        ],
        parameters={
            'SDate': start_date,
            'EDate': end_date,
            'Frq': 'D'
        }
    )

    splits, splits_err = ek.get_data(
        instruments=assets,
        fields=['TR.CAEffectiveDate', 'TR.CAAdjustmentFactor'],
        parameters={
            "CAEventType": "SSP",
            'SDate': start_date,
            'EDate': end_date,
            'Frq': 'D'
        }
    )

    prices.rename(
        columns={
            'Open Price': 'open',
            'High Price': 'high',
            'Low Price': 'low',
            'Close Price': 'close'
        },
        inplace=True
    )
    prices.dropna(inplace=True)
    prices['Date'] = pd.to_datetime(prices['Date']).dt.date

    divs.rename(
        columns={
            'Dividend Ex Date': 'Date',
            'Gross Dividend Amount': 'div_amt',
            'Dividend Type': 'div_type',
            'Dividend Payment Type': 'pay_type'
        },
        inplace=True
    )
    divs.dropna(inplace=True)
    divs['Date'] = pd.to_datetime(divs['Date']).dt.date
    divs = divs[(divs.Date.notnull()) & (divs.div_amt > 0)]

    splits.rename(
        columns={
            'Capital Change Effective Date': 'Date',
            'Adjustment Factor': 'split_rto'
        },
        inplace=True
    )
    splits.dropna(inplace=True)
    splits['Date'] = pd.to_datetime(splits['Date']).dt.date

    unadjusted_price_history = pd.merge(
        prices, divs[['Instrument', 'Date', 'div_amt']],
        how='outer',
        on=['Date', 'Instrument']
    )
    unadjusted_price_history['div_amt'].fillna(0, inplace=True)

    unadjusted_price_history = pd.merge(
        unadjusted_price_history, splits,
        how='outer',
        on=['Date', 'Instrument']
    )
    unadjusted_price_history['split_rto'].fillna(1, inplace=True)

    if unadjusted_price_history.isnull().values.any():
        raise Exception('missing values detected!')
    history = unadjusted_price_history.to_dict('records')
    return history

@app.callback(
    Output("returns-tbl", "data"),
    Input("history-tbl", "data"),
    prevent_initial_call = True
)
def calculate_returns(history_tbl):

    dt_prc_div_splt = pd.DataFrame(history_tbl)

    # Define what columns contain the Identifier, date, price, div, & split info
    ins_col = 'Instrument'
    dte_col = 'Date'
    prc_col = 'close'
    div_col = 'div_amt'
    spt_col = 'split_rto'

    dt_prc_div_splt[dte_col] = pd.to_datetime(dt_prc_div_splt[dte_col])
    dt_prc_div_splt = dt_prc_div_splt.sort_values([ins_col, dte_col])[
        [ins_col, dte_col, prc_col, div_col, spt_col]].groupby(ins_col)
    numerator = dt_prc_div_splt[[dte_col, ins_col, prc_col, div_col]].tail(-1)
    denominator = dt_prc_div_splt[[prc_col, spt_col]].head(-1)

    res = pd.DataFrame({
        'Date': numerator[dte_col].reset_index(drop=True),
        'Instrument': numerator[ins_col].reset_index(drop=True),
        'rtn': np.log(
            (numerator[prc_col] + numerator[div_col]).reset_index(drop=True) / (
                    denominator[prc_col] * denominator[spt_col]
            ).reset_index(drop=True)
        )
    }).pivot_table(
            values='rtn', index='Date', columns='Instrument'
        ).reset_index()
    res['Date'] = pd.to_datetime(res['Date'].dt.date)
    return(res.to_dict('records')
    )

@app.callback(
    Output("new-date-range", "min_date_allowed"),
    Output("new-date-range", "max_date_allowed"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    prevent_initial_call=True
)
def time_adjustment(start, end):
    new_start = pd.to_datetime(start) + pd.Timedelta(days=1)
    return new_start.strftime("%Y-%M-%d"), end
# @app.callback(
#     Output("history-tbl", "data"),
#     Input("run-query-plot", "n_clicks"),
#     [State('benchmark-id', 'value'), State('asset-id', 'value'), State('date-range', 'start_date'), State('date-range', 'end_date')],
#     prevent_initial_call=True
# )

@app.callback(
    Output("ab-plot", "figure"),
    Input("returns-tbl", "data"),
    [State('benchmark-id', 'value'), State('asset-id', 'value'),
    State('new-date-range', 'start_date'), State('new-date-range', 'end_date')],
    prevent_initial_call = True
)
def render_ab_plot(returns, benchmark_id, asset_id, start, end):
    rtn = pd.DataFrame(returns)
    rtn['Date'] = pd.to_datetime(rtn['Date']).dt.date
    rtn = rtn[(rtn['Date'] >= pd.Timestamp(start)) & (rtn['Date'] <= pd.Timestamp(end))]
    return(
        px.scatter(rtn, x=benchmark_id, y=asset_id, trendline='ols')
    )

@app.callback(
    Output("alpha-beta-output", "children"),
    Input("returns-tbl", "data"),
    [State('benchmark-id', 'value'), State('asset-id', 'value'),
    State('new-date-range', 'start_date'), State('new-date-range', 'end_date')],
    prevent_initial_call = True
)
def print_out_alphabeta(returns,benchmark_id,asset_id, start, end):
    rtn = pd.DataFrame(returns)
    rtn['Date'] = pd.to_datetime(rtn['Date']).dt.date
    rtn = rtn[(rtn['Date'] >= pd.Timestamp(start)) & (rtn['Date'] <= pd.Timestamp(end))]
    plot = px.scatter(rtn,x=benchmark_id,y=asset_id,trendline='ols')
    model = px.get_trendline_results(plot)
    alpha = model.iloc[0]["px_fit_results"].params[0]
    beta = model.iloc[0]["px_fit_results"].params[1]
    return f"Alpha: {alpha} | Beta: {beta}"


@app.callback(
    Output("entry-blotter", "data"),
    Input("alpha-beta-output", "children"),
    prevent_initial_call = True
)
def entry_blotter_generation(alpha):
    start_date_str = '2023-01-30'
    end_date_str = '2023-02-08'

    ivv_prc, ivv_prc_err = ek.get_data(
        instruments = ["IVV"],
        fields = [
            'TR.OPENPRICE(Adjusted=0)',
            'TR.HIGHPRICE(Adjusted=0)',
            'TR.LOWPRICE(Adjusted=0)',
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters = {
            'SDate': start_date_str,
            'EDate': end_date_str,
            'Frq': 'D'
        }
    )

    ivv_prc['Date'] = pd.to_datetime(ivv_prc['Date']).dt.date
    ivv_prc.drop(columns='Instrument', inplace=True)

    ##### Get the next business day from Refinitiv!!!!!!!
    rd.open_session()

    next_business_day = rd.dates_and_calendars.add_periods(
        start_date=ivv_prc['Date'].iloc[-1].strftime("%Y-%m-%d"),
        period="1D",
        calendars=["USA"],
        date_moving_convention="NextBusinessDay",
    )

    rd.close_session()
    ######################################################

    alpha1 = -0.01
    n1 = 3

    # submitted entry orders
    submitted_entry_orders = pd.DataFrame({
        "trade_id": range(1, ivv_prc.shape[0]),
        "date": list(pd.to_datetime(ivv_prc["Date"].iloc[1:]).dt.date),
        "asset": "IVV",
        "trip": 'ENTER',
        "action": "BUY",
        "type": "LMT",
        "price": round(
            ivv_prc['Close Price'].iloc[:-1] * (1 + alpha1),
            2
        ),
        'status': 'SUBMITTED'
    })

    # if the lowest traded price is still higher than the price you bid, then the
    # order never filled and was cancelled.
    with np.errstate(invalid='ignore'):
        cancelled_entry_orders = submitted_entry_orders[
            np.greater(
                ivv_prc['Low Price'].iloc[1:][::-1].rolling(3).min()[::-1].to_numpy(),
                submitted_entry_orders['price'].to_numpy()
            )
        ].copy()
    cancelled_entry_orders.reset_index(drop=True, inplace=True)
    cancelled_entry_orders['status'] = 'CANCELLED'
    cancelled_entry_orders['date'] = pd.DataFrame(
        {'cancel_date': submitted_entry_orders['date'].iloc[(n1 - 1):].to_numpy()},
        index=submitted_entry_orders['date'].iloc[:(1 - n1)].to_numpy()
    ).loc[cancelled_entry_orders['date']]['cancel_date'].to_list()
    # print(cancelled_entry_orders)

    filled_entry_orders = submitted_entry_orders[
        submitted_entry_orders['trade_id'].isin(
            list(
                set(submitted_entry_orders['trade_id']) - set(
                    cancelled_entry_orders['trade_id']
                )
            )
        )
    ].copy()
    filled_entry_orders.reset_index(drop=True, inplace=True)
    filled_entry_orders['status'] = 'FILLED'
    for i in range(0, len(filled_entry_orders)):

        idx1 = np.flatnonzero(
            ivv_prc['Date'] == filled_entry_orders['date'].iloc[i]
        )[0]

        ivv_slice = ivv_prc.iloc[idx1:(idx1 + n1)]['Low Price']

        fill_inds = ivv_slice <= filled_entry_orders['price'].iloc[i]

        if (len(fill_inds) < n1) & (not any(fill_inds)):
            filled_entry_orders.at[i, 'status'] = 'LIVE'
        else:
            filled_entry_orders.at[i, 'date'] = ivv_prc['Date'].iloc[
                fill_inds.idxmax()
            ]

    live_entry_orders = pd.DataFrame({
        "trade_id": ivv_prc.shape[0],
        "date": pd.to_datetime(next_business_day).date(),
        "asset": "IVV",
        "trip": 'ENTER',
        "action": "BUY",
        "type": "LMT",
        "price": round(ivv_prc['Close Price'].iloc[-1] * (1 + alpha1), 2),
        'status': 'LIVE'
    },
        index=[0]
    )

    if any(filled_entry_orders['status'] == 'LIVE'):
        live_entry_orders = pd.concat([
            filled_entry_orders[filled_entry_orders['status'] == 'LIVE'],
            live_entry_orders
        ])
        live_entry_orders['date'] = pd.to_datetime(next_business_day).date()

    filled_entry_orders = filled_entry_orders[
        filled_entry_orders['status'] == 'FILLED'
        ]

    entry_orders = pd.concat(
        [
            submitted_entry_orders,
            cancelled_entry_orders,
            filled_entry_orders,
            live_entry_orders
        ]
    ).sort_values(["date", 'trade_id'])
    entry = entry_orders.to_dict('records')
    return entry


@app.callback(
    Output("exit-blotter", "data"),
    Input("alpha-beta-output", "children"),
    prevent_initial_call = True
)
def exit_blotter_generation(alpha):
    start_date_str = '2023-01-30'
    end_date_str = '2023-02-08'

    ivv_prc, ivv_prc_err = ek.get_data(
        instruments = ["IVV"],
        fields = [
            'TR.OPENPRICE(Adjusted=0)',
            'TR.HIGHPRICE(Adjusted=0)',
            'TR.LOWPRICE(Adjusted=0)',
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters = {
            'SDate': start_date_str,
            'EDate': end_date_str,
            'Frq': 'D'
        }
    )

    ivv_prc['Date'] = pd.to_datetime(ivv_prc['Date']).dt.date
    ivv_prc.drop(columns='Instrument', inplace=True)

    ##### Get the next business day from Refinitiv!!!!!!!
    rd.open_session()

    next_business_day = rd.dates_and_calendars.add_periods(
        start_date=ivv_prc['Date'].iloc[-1].strftime("%Y-%m-%d"),
        period="1D",
        calendars=["USA"],
        date_moving_convention="NextBusinessDay",
    )

    rd.close_session()
    ######################################################

    alpha2 = 0.01
    n2 = 5

    # submitted sell orders
    submitted_exit_orders = pd.DataFrame({
        "trade_id": range(1, ivv_prc.shape[0]),
        "date": list(pd.to_datetime(ivv_prc["Date"].iloc[1:]).dt.date),
        "asset": "IVV",
        "trip": 'EXIT',
        "action": "SELL",
        "type": "LMT",
        "price": round(
            ivv_prc['Close Price'].iloc[:-1] * (1 + alpha2),
            2
        ),
        'status': 'SUBMITTED'
    })

    # if the highest traded price is still lower than the price you bid, then the
    # order never filled and was cancelled.
    with np.errstate(invalid='ignore'):
        cancelled_exit_orders = submitted_exit_orders[
            np.less(
                ivv_prc['High Price'].iloc[1:][::-1].rolling(n2).max()[::-1].to_numpy(),
                submitted_exit_orders['price'].to_numpy()
            )
        ].copy()
    cancelled_exit_orders.reset_index(drop=True, inplace=True)
    cancelled_exit_orders['status'] = 'CANCELLED'
    cancelled_exit_orders['date'] = pd.DataFrame(
        {'cancel_date': submitted_exit_orders['date'].iloc[(n2 - 1):].to_numpy()},
        index=submitted_exit_orders['date'].iloc[:(1 - n2)].to_numpy()
    ).loc[cancelled_exit_orders['date']]['cancel_date'].to_list()
    # print(cancelled_exit_orders)

    filled_exit_orders = submitted_exit_orders[
        submitted_exit_orders['trade_id'].isin(
            list(
                set(submitted_exit_orders['trade_id']) - set(
                    cancelled_exit_orders['trade_id']
                )
            )
        )
    ].copy()
    filled_exit_orders.reset_index(drop=True, inplace=True)
    filled_exit_orders['status'] = 'FILLED'
    for i in range(0, len(filled_exit_orders)):

        idx1 = np.flatnonzero(
            ivv_prc['Date'] == filled_exit_orders['date'].iloc[i]
        )[0]

        ivv_slice = ivv_prc.iloc[idx1:(idx1 + n2)]['Low Price']

        fill_inds = ivv_slice <= filled_exit_orders['price'].iloc[i]

        if (len(fill_inds) < n2) & (not any(fill_inds)):
            filled_exit_orders.at[i, 'status'] = 'LIVE'
        else:
            filled_exit_orders.at[i, 'date'] = ivv_prc['Date'].iloc[
                fill_inds.idxmax()
            ]

    live_exit_orders = pd.DataFrame({
        "trade_id": ivv_prc.shape[0],
        "date": pd.to_datetime(next_business_day).date(),
        "asset": "IVV",
        "trip": 'EXIT',
        "action": "SELL",
        "type": "LMT",
        "price": round(ivv_prc['Close Price'].iloc[-1] * (1 + alpha2), 2),
        'status': 'LIVE'
    },
        index=[0]
    )

    if any(filled_exit_orders['status'] == 'LIVE'):
        live_exit_orders = pd.concat([
            filled_exit_orders[filled_exit_orders['status'] == 'LIVE'],
            live_exit_orders
        ])
        live_exit_orders['date'] = pd.to_datetime(next_business_day).date()

    filled_exit_orders = filled_exit_orders[
        filled_exit_orders['status'] == 'FILLED'
        ]

    exit_orders = pd.concat(
        [
            submitted_exit_orders,
            cancelled_exit_orders,
            filled_exit_orders,
            live_exit_orders
        ]
    ).sort_values(["date", 'trade_id'])
    exit = exit_orders.to_dict('records')
    return exit

if __name__ == '__main__':
    app.run_server(debug=True, port=8000)
