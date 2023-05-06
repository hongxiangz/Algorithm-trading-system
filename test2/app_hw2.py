from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, date, timedelta
import plotly.express as px
import pandas as pd
from datetime import datetime
import numpy as np
import os
import refinitiv.dataplatform.eikon as ek
import refinitiv.dataplatform as rd

ek.set_app_key(os.getenv('EIKON_API'))

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

percentage = dash_table.FormatTemplate.percentage(3)

controls = dbc.Card(
    [
        dbc.Row(html.Button('QUERY Refinitiv', id='run-query', n_clicks=0)),
        dbc.Row([
            html.H5('Asset:',
                    style={'display': 'inline-block', 'margin-right': 20}),
            dcc.Input(id='asset', type='text', value="IVV",
                      style={'display': 'inline-block',
                             'border': '1px solid black'}),
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("α1"), html.Th("n1")])),
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                dbc.Input(
                                    id='alpha1',
                                    type='number',
                                    value=-0.01,
                                    max=1,
                                    min=-1,
                                    step=0.01
                                )
                            ),
                            html.Td(
                                dcc.Input(
                                    id='n1',
                                    type='number',
                                    value=3,
                                    min=1,
                                    step=1
                                )
                            )
                        ])
                    ])
                ],
                bordered=True
            ),
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("α2"), html.Th("n2")])),
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                dbc.Input(
                                    id='alpha2',
                                    type='number',
                                    value=0.01,
                                    max=1,
                                    min=-1,
                                    step=0.01
                                )
                            ),
                            html.Td(
                                dcc.Input(
                                    id='n2',
                                    type='number',
                                    value=5,
                                    min=1,
                                    step=1
                                )
                            )
                        ])
                    ])
                ],
                bordered=True
            )
        ]),
        dbc.Row([
            dcc.DatePickerRange(
                id='refinitiv-date-range',
                min_date_allowed = date(2015, 1, 1),
                max_date_allowed = datetime.now(),
                initial_visible_month = date(2023,2,1),
                start_date = datetime.date(
                    datetime.now() - timedelta(days=3*365)
                ),
                end_date = datetime.now().date()
            )
        ])
    ],
    body=True
)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(
                    # Put your reactive graph here as an image!
                    html.Img(src='reactive-graph.png',style={'width':'100%'}),
                    md = 8
                )
            ],
            align="center",
        ),
        html.H2('Trade Blotter:'),
        dash_table.DataTable(id = "blotter")
    ],
    fluid=True
)

@app.callback(
    Output("blotter", "data"),
    Input("run-query", "n_clicks"),
    [State('asset', 'value'), State('refinitiv-date-range', 'start_date'), State('refinitiv-date-range', 'end_date'),
     State('alpha1','value'), State('n1', 'value'), State('alpha2','value'), State('n2','value')],
    prevent_initial_call=True
)
def search_data(n_clicks, asset, start_date, end_date, a1, gn1, a2, gn2):
    start_date_str = start_date
    end_date_str = end_date

    ivv_prc, ivv_prc_err = ek.get_data(
        instruments = [asset],
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
        start_date= ivv_prc['Date'].iloc[-1].strftime("%Y-%m-%d"),
        period="1D",
        calendars=["USA"],
        date_moving_convention="NextBusinessDay",
    )

    rd.close_session()
    ######################################################
    alpha1 = a1
    n1 = gn1

    alpha2 = a2
    n2 = gn2
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
    # ivv_prc1 = ivv_prc
    with np.errstate(invalid='ignore'):
        cancelled_entry_orders = submitted_entry_orders[
            np.greater(
                ivv_prc['Low Price'].iloc[1:][::-1].rolling(n1).min()[::-1].to_numpy(),
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

    submitted_exit_orders = filled_entry_orders.copy()
    submitted_exit_orders['trip'] = 'EXIT'
    submitted_exit_orders['action'] = 'SELL'
    submitted_exit_orders['status'] = 'SUBMITTED'
    submitted_exit_orders['price'] = filled_entry_orders['price']*(1+alpha2)

    # live_exit_orders = submitted_exit_orders.copy()
    # live_exit_orders['status'] = 'LIVE'

    # temp_exit_orders = submitted_entry_orders.copy()

    # for i in temp_exit_orders['trade_id']:
    #     if i in list(submitted_exit_orders['trade_id']):
    #         temp_exit_orders.loc[temp_exit_orders['trade_id']==i, 'price'] = submitted_exit_orders.loc[submitted_exit_orders['trade_id']==i,'price'].values[0]

    cancelled_price = list(ivv_prc['High Price'].iloc[1:][::-1].rolling(n2).max()[::-1].to_numpy())
    date_list = list(submitted_entry_orders['date'])
    exit_date = list(submitted_exit_orders['date'])
    lst = []
    for i in range(len(date_list)):
        if date_list[i] in exit_date:
            lst.append(i)
    cancelled_exit_orders = pd.DataFrame()
    for i in range(len(lst)):
        if not np.isnan(cancelled_price[lst[i]]):
            if float(submitted_exit_orders.loc[submitted_exit_orders['date'] == exit_date[i], 'price']) > cancelled_price[lst[i]]:
                cancelled_exit_orders = pd.concat([cancelled_exit_orders, submitted_exit_orders.loc[submitted_exit_orders['date'] == exit_date[i]]]).copy()
        # if np.isnan(cancelled_price[lst[i]]):
        #     if cancelled_exit_orders.shape[0] == 0:
        #         cancelled_exit_orders = submitted_exit_orders.loc[submitted_exit_orders['date'] == exit_date[i]]
        #     else:
        #         cancelled_exit_orders = pd.concat([cancelled_exit_orders, submitted_exit_orders.loc[
        #             submitted_exit_orders['date'] == exit_date[i]]]).copy
        # else:
        #     if float(submitted_exit_orders.loc[submitted_exit_orders['date'] == exit_date[i], 'price']) > cancelled_price[lst[i]]:
        #         if cancelled_exit_orders.shape[0] == 0:
        #             cancelled_exit_orders = submitted_exit_orders.loc[submitted_exit_orders['date'] == exit_date[i]]
        #         else:
        #             cancelled_exit_orders = pd.concat([cancelled_exit_orders, submitted_exit_orders.loc[
        #                 submitted_exit_orders['date'] == exit_date[i]]]).copy
    # cancelled_exit_orders['status'] = 'CANCELLED'

    # if the highest traded price is still lower than the price you bid, then the
    # order never filled and was cancelled.
    # with np.errstate(invalid='ignore'):
    #     cancelled_exit_orders = submitted_exit_orders[
    #         np.greater(
    #             submitted_exit_orders['price'].to_numpy(),
    #             ivv_prc['High Price'].iloc[1:][::-1].rolling(n2).max()[::-1].to_numpy()
    #
    #         )
    #     ].copy()

    cancelled_exit_orders.reset_index(drop=True, inplace=True)
    cancelled_exit_orders['status'] = 'CANCELLED'
    datelist = list(pd.to_datetime((cancelled_exit_orders['date'])))
    cancelled_date_list = []
    for i in range(len(datelist)):
        daterange = pd.bdate_range(start=datelist[i], end=datelist[i]+timedelta(n2))
        cancelled_date_list.append(daterange[-1])
    cancelled_exit_orders['date'] = cancelled_date_list
    cancelled_exit_orders['date'] = cancelled_exit_orders['date'].dt.strftime('%Y-%m-%d')

    # filled_exit_orders = submitted_exit_orders[
    #     submitted_exit_orders['trade_id'].isin(
    #         list(
    #             set(submitted_exit_orders['trade_id']) - set(
    #                 cancelled_exit_orders['trade_id']
    #             )
    #         )
    #     )
    # ].copy()
    # filled_exit_orders.reset_index(drop=True, inplace=True)
    # filled_exit_orders['status'] = 'FILLED'
    # for i in range(0, len(filled_exit_orders)):
    #
    #     idx1 = np.flatnonzero(
    #         ivv_prc['Date'] == filled_exit_orders['date'].iloc[i]
    #     )[0]
    #
    #     ivv_slice = ivv_prc.iloc[idx1:(idx1 + n2)]['Low Price']
    #
    #     fill_inds = ivv_slice <= filled_exit_orders['price'].iloc[i]
    #
    #     if (len(fill_inds) < n2) & (not any(fill_inds)):
    #         filled_exit_orders.at[i, 'status'] = 'LIVE'
    #     else:
    #         filled_exit_orders.at[i, 'date'] = ivv_prc['Date'].iloc[
    #             fill_inds.idxmax()
    #         ]

    # filled_entry_orders['status'] = 'FILLED'
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

        ivv_slice = ivv_prc.iloc[idx1:(idx1 + n2)]['High Price']

        fill_inds = ivv_slice >= filled_exit_orders['price'].iloc[i]

        if (len(fill_inds) < n2) & (not any(fill_inds)):
            filled_exit_orders.at[i, 'status'] = 'LIVE'
        else:
            filled_exit_orders.at[i, 'date'] = ivv_prc['Date'].iloc[
                fill_inds.idxmax()
            ]

    # live_exit_orders = pd.DataFrame({
    #     "trade_id": ivv_prc.shape[0],
    #     "date": pd.to_datetime(next_business_day).date(),
    #     "asset": "IVV",
    #     "trip": 'EXIT',
    #     "action": "SELL",
    #     "type": "LMT",
    #     "price": round(ivv_prc['Close Price'].iloc[-1] * (1 + alpha2), 2),
    #     'status': 'LIVE'
    # },
    #     index=[0]
    # )
    live_exit_orders = pd.DataFrame()
    if any(filled_exit_orders['status'] == 'LIVE'):
        live_exit_orders = pd.concat([
            filled_exit_orders[filled_exit_orders['status'] == 'LIVE'],
            live_exit_orders
        ])
        live_exit_orders['date'] = pd.to_datetime(next_business_day).date()

    filled_exit_orders = filled_exit_orders[
        filled_exit_orders['status'] == 'FILLED'
        ]

    # if any(filled_exit_orders['status'] == 'LIVE'):
    #     live_exit_orders = pd.concat([
    #         filled_exit_orders[filled_exit_orders['status'] == 'LIVE'],
    #         live_exit_orders
    #     ])
    #     live_exit_orders['date'] = pd.to_datetime(next_business_day).date()
    #
    # filled_exit_orders = filled_exit_orders[
    #     filled_exit_orders['status'] == 'FILLED'
    #     ]

    entry_orders = pd.concat(
        [
            submitted_entry_orders,
            cancelled_entry_orders,
            filled_entry_orders,
            live_entry_orders,
            submitted_exit_orders,
            cancelled_exit_orders,
            filled_exit_orders,
            live_exit_orders
        ]
    ).sort_values(['trade_id', 'action','date'])
    entry = entry_orders.to_dict('records')
    return entry


if __name__ == '__main__':
    app.run_server(debug=True, port=8020)