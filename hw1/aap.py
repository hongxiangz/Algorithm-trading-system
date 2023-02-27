from dash import Dash, html, dcc, dash_table, Input, Output, State
import eikon as ek
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.express as px
import os
import statsmodels

app = Dash(__name__)

ek.set_app_key(os.getenv('EIKON_API'))

# spacer = html.Div(style={'margin': '10px','display':'inline'})
assets = ['AAPL.O', 'IVV', 'GLD', 'SHY.O', "MSFT.O", "TSLA.O"]

app.layout = html.Div(
    [
        # Choosing Assets
        html.Div(
            [
                html.Div(
                    [
                        html.Label('Benchmark'),
                        dcc.Dropdown(assets, assets[0], id='benchmark-id')
                    ],
                    style={'padding': 10, 'flex': 1, 'width': '200px'}
                ),
                html.Div(
                    [
                        html.Label('Asset'),
                        dcc.Dropdown(assets, assets[1], id='asset-id')
                    ],
                    style={'padding': 10, 'flex': 1, 'width': '200px'}
                )
            ],
            style={'display': 'flex', 'flex-direction': 'row'}
        ),
        # Choosing dates & Query button
        html.Div(
            [
                html.Label('Start date/End date'),
                html.Br(),
                dcc.DatePickerRange(
                    id='my-date-picker-range',
                    start_date=date(2020, 1, 1),
                    end_date=datetime.now().strftime("%m/%d/%Y"),
                    calendar_orientation='vertical',
                    max_date_allowed=datetime.now(),
                    month_format='YYYY-MM-DD',
                    style={'font-size': '14px'}
                ),
                "\t",
                html.Button('Query Refinitiv', id='run-query', n_clicks=0,
                            style={"height": "30px", "width": "200px", "font-size": "14px"}),
            ],
            style={'padding': 10, 'flex': 1}
        ),
        # Raw data table
        html.Br(),
        html.H2('Raw Data from Refinitiv'),
        dash_table.DataTable(
            id="history-tbl",
            page_action='none',
            style_table={'height': '300px', 'overflowY': 'auto', 'overflowX': 'scroll'}
        ),
        # Return table
        html.Br(),
        html.H2('Historical Returns'),
        dash_table.DataTable(
            id="returns-tbl",
            page_action='none',
            style_table={'height': '300px', 'overflowY': 'auto', 'overflowX': 'scroll'}
        ),
        # "Scatter plot choosing dates", "Query button" and "Alpha & Beta output"
        html.Br(),
        html.H2('Alpha & Beta Scatter Plot'),
        html.Div(
            [
                html.Label('Start date/End date'),
                html.Br(),
                dcc.DatePickerRange(
                    id='my-date-picker-range-plot',
                    start_date=date(2021, 1, 1),
                    end_date=datetime.now().strftime("%m/%d/%Y"),
                    calendar_orientation='vertical',
                    max_date_allowed=datetime.now(),
                    month_format='YYYY-MM-DD',
                    style={'font-size': '14px'}
                ),
                "\t",
                html.Button('Update date', id='update-plot', n_clicks=0,
                            style={"height": "30px", "width": "200px", "font-size": "14px"}),
                "\t",
                html.Label('Alpha:',  style={"font-size": "18px"}),
                html.Output('1'),
                html.Label('Beta:',  style={"font-size": "18px"}),
                html.Output('2')
            ],
            style={'padding': 10, 'flex': 1}
        ),

        # Scatter plot
        dcc.Graph(id="ab-plot"),
        html.P(id='summary-text', children="")
    ]
)


@app.callback(
    Output("history-tbl", "data"),
    Input("run-query", "n_clicks"),
    [State('benchmark-id', 'value'), State('asset-id', 'value'),
     State('my-date-picker-range', 'start_date'), State('my-date-picker-range', 'end_date')],
    prevent_initial_call=True
)
def query_refinitiv(n_clicks, benchmark_id, asset_id, start_date, end_date):
    assets = [benchmark_id, asset_id]
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

    return unadjusted_price_history.to_dict('records')


@app.callback(
    Output("returns-tbl", "data"),
    Input("history-tbl", "data"),
    prevent_initial_call=True
)
def calculate_returns( history_tbl):

    dt_prc_div_splt = pd.DataFrame(history_tbl)

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

    pivot = pd.DataFrame({
        'Date': numerator[dte_col].reset_index(drop=True),
        'Instrument': numerator[ins_col].reset_index(drop=True),
        'rtn': np.log(
            (numerator[prc_col] + numerator[div_col]).reset_index(drop=True) / (
                    denominator[prc_col] * denominator[spt_col]
            ).reset_index(drop=True)
        )
    }).pivot_table(
            values='rtn', index='Date', columns='Instrument'
        )
    pivot["Date"] = pd.to_datetime(pivot.index).strftime("%Y-%m-%d")
    return pivot.to_dict('records')


@app.callback(
    Output("ab-plot", "figure"),
    Input("update-plot", "n_clicks"),
    Input("returns-tbl", "data"),
    [State('benchmark-id', 'value'), State('asset-id', 'value'),
     State('my-date-picker-range-plot', 'start_date'), State('my-date-picker-range-plot', 'end_date')],
    prevent_initial_call=True
)
def render_ab_plot(n_clicks, returns, benchmark_id, asset_id, start_date, end_date):

    returns = pd.DataFrame(returns)

    if start_date is not None:
        if end_date is not None:
            returns = returns.loc[(returns['Date'] >= start_date) & (returns['Date'] <= end_date)]

    return (
        px.scatter(returns, x=benchmark_id, y=asset_id, trendline='ols')
    )

# @app.callback(
#     Output("ab-tbl", "data"),
#     Input("ab-plot", "figure"),
#     prevent_initial_call=True
# )
# def render_ab_tbl(ab_plot):
#
#     trend_line = ab_plot['data'][1]['hovertemplate']
#     regression_equation_elements_list = trend_line.split("<br>")[1].strip().split()
#     beta,risk_free_rtn = float(regression_equation_elements_list[2]), float(regression_equation_elements_list[6])
#
#     asset_return = ab_plot['data'][1]['y']
#     asset_avg_rtn = sum(asset_return)/len(asset_return)
#     benchmark_return = ab_plot['data'][1]['x']
#     mkt_avg_rtn = sum(benchmark_return)/len(benchmark_return)
#     alpha = asset_avg_rtn - risk_free_rtn - beta * (mkt_avg_rtn-risk_free_rtn)
#
#     return (
#         pd.DataFrame({'alpha':[alpha],'beta': [beta]}).to_dict('records')
#     )

    # plot = px.scatter(returns, x=benchmark_id, y=asset_id, trendline='ols')
    # model = px.get_trendline_results(plot)
    # results = model.iloc[0]["px_fit_results"]
    # alpha = results.params[0]
    # beta = results.params[1]



if __name__ == '__main__':
    app.run_server(debug=True)
