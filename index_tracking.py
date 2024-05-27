import yahoo_fin.stock_info as si
import yfinance as yf

import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

from requests_html import HTMLSession

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import numpy as np


# # FUNCTIONS FOR YFINANCE :
# def get_ticker_data_yfinance(ticker, n_months):
#     hist = yf.Ticker(ticker).history(period=str(n_months) + "mo")
#     return hist.Close


# def get_stocks_data_yfinance(tickers_list, n_months):
#     stocks_historical = {}
#     for ticker in tickers_list:
#         try:
#             stocks_historical[ticker] = get_ticker_data_yfinance(ticker, n_months)
#         except AssertionError as error:
#             print(ticker, error)
#             tickers_list.remove(ticker)
#             stocks_historical.pop(ticker)

#     return stocks_historical  # Returns a dictionary of stocks series


# FUNCTIONS FOR YFINANCE :
def get_ticker_data_yfinance(ticker, start_date, end_date):
    hist = yf.download(ticker, start=start_date, end=end_date)
    return hist.Close


def get_stocks_data_yfinance(tickers_list, start_date, end_date):
    stocks_historical = {}
    for ticker in tickers_list:
        try:
            stocks_historical[ticker] = get_ticker_data_yfinance(
                ticker, start_date, end_date
            )
        except AssertionError as error:
            print(ticker, error)
            tickers_list.remove(ticker)
            stocks_historical.pop(ticker)

    return stocks_historical  # Returns a dictionary of stocks series


# FUNCTIONS FOR YAHOO_FIN :
def get_ticker_data_yahoofin(ticker, start_date, end_date):
    temp_df = si.get_data(
        ticker, start_date=start_date, end_date=end_date, interval="1d"
    )
    return temp_df.close


def get_stocks_data_yahoofin(tickers_list, start_date, end_date):
    stocks_historical = {}
    for ticker in tickers_list:
        try:
            stocks_historical[ticker] = get_ticker_data_yahoofin(
                ticker, start_date, end_date
            )
        except AssertionError as error:
            print(ticker, error)
            tickers_list.remove(ticker)
            stocks_historical.pop(ticker)

    return stocks_historical  # Returns a dictionary of stocks series


# FUNCTIONS FOR GUROBI :
def create_gurobi_model(tickers_list, index_historical, stocks_historical, K):
    time_period = len(tickers_list)

    m = gp.Model("index_tracking_model")

    # Decision Variables :
    z = pd.Series(
        m.addVars(tickers_list, vtype=GRB.BINARY, name=tickers_list), index=tickers_list
    )
    w = pd.Series(
        m.addVars(
            tickers_list, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=tickers_list
        ),
        index=tickers_list,
    )

    # Constraints :
    m.addConstr(gp.quicksum(z[ticker] for ticker in tickers_list) <= K, "c0")
    m.addConstrs((w[ticker] <= z[ticker] for ticker in tickers_list), name="c1")
    m.addConstr(gp.quicksum(w[ticker] for ticker in tickers_list) == 1.0, "c2")

    # Objective Function :
    gurobi_function = create_objective_function(
        tickers_list, index_historical, stocks_historical, time_period, w
    )

    m.setObjective(gurobi_function, GRB.MINIMIZE)
    m.update()
    m.optimize()

    return get_weights(w)


def create_objective_function(
    tickers_list, index_historical, stocks_historical, time_period, w
):
    qexpr = gp.QuadExpr(0)
    last_date = None
    # time_period = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

    for date in index_historical.index:
        if last_date != None:
            lexpr = gp.LinExpr(0)
            for ticker in tickers_list:
                lexpr.add(
                    w[ticker],
                    (
                        stocks_historical[ticker][date]
                        / stocks_historical[ticker][last_date]
                        - 1
                    ),
                )
            lexpr.addConstant(
                -(index_historical[date] / index_historical[last_date] - 1)
            )
            qexpr.add(lexpr * lexpr)
        last_date = date

    return qexpr / time_period


def get_weights(w):
    w_dif_zero = {}
    for i in w:
        if i.X != 0:
            w_dif_zero[i.VarName] = i.X
    return w_dif_zero


def get_index_return(index_historical):
    index_return = []
    last_date = None
    for date in index_historical.index:
        if last_date != None:
            index_return.append(
                index_historical[date] / index_historical[last_date] - 1
            )
        last_date = date
    return index_return


def get_all_selected_stocks_return(stocks_historical, weights, index_historical):
    def get_stock_return(ticker, stocks_historical):
        temp_return = []
        last_date = None
        for date in index_historical.index:
            if last_date != None:
                temp_return.append(
                    stocks_historical[ticker][date]
                    / stocks_historical[ticker][last_date]
                    - 1
                )
            last_date = date
        return temp_return

    selected_stocks_returns = []
    i = 0
    for dict_key in weights.keys():
        selected_stocks_returns.append(get_stock_return(dict_key, stocks_historical))

        weight = weights[dict_key]
        for j in range(len(selected_stocks_returns[i])):
            selected_stocks_returns[i][j] = weight * selected_stocks_returns[i][j]
        i = i + 1
    selected_stocks_returns_df = pd.DataFrame(selected_stocks_returns)

    selected_stocks_mean_return = []
    for i in range(selected_stocks_returns_df.shape[1]):
        selected_stocks_mean_return.append(selected_stocks_returns_df[i].sum())

    return selected_stocks_mean_return


def get_all_selected_stocks_accumulated_return(
    stocks_historical, weights, index_historical
):
    def get_stock_return(ticker, stocks_historical):
        temp_return = []
        last_date = None
        for date in index_historical.index:
            if last_date != None:
                temp_return.append(
                    stocks_historical[ticker][date]
                    / stocks_historical[ticker][last_date]
                    - 1
                )
            last_date = date
        return temp_return

    selected_stocks_returns = {}
    for dict_key in weights.keys():
        selected_stocks_returns[dict_key] = get_stock_return(
            dict_key, stocks_historical
        )

    start_value = index_historical[index_historical.index.min()]
    days = list(index_historical.keys())
    tickers = list(weights.keys())

    stock_values = np.zeros((len(tickers), len(days) - 1))

    for i in range(len(days) - 1):  # days
        day_total_value = 0
        for j in range(len(tickers)):  # tickers
            if i == 0:
                stock_values[j][i] = weights[tickers[j]] * start_value
            else:
                stock_values[j][i] = stock_values[j][i - 1] * (
                    1 + selected_stocks_returns[tickers[j]][i]
                )

            day_total_value += stock_values[j][i]

    days.pop(0)
    return pd.Series(np.sum(stock_values, axis=0), index=days)


def get_sp100_tickers():
    req = HTMLSession()
    r = req.get(
        r"https://en.wikipedia.org/wiki/S%26P_100",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    temp_df = pd.read_html(r.text, attrs={"id": "constituents"})
    return list(temp_df[0].Symbol)


def get_ibovespa_top30_tickers():
    req = HTMLSession()
    r = req.get(
        r"https://finance.yahoo.com/quote/%5EBVSP/components",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    temp_df = pd.read_html(r.text, index_col="Symbol")[0]
    return temp_df.index.to_list()


def get_IBOV_from_B3():
    # Create a browser instance
    driver = webdriver.Chrome()

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run Chrome in headless mode
    driver = webdriver.Chrome(options=options)

    # Open a web page
    driver.get(
        "https://sistemaswebb3-listados.b3.com.br/indexPage/day/IBOV?language=en-us"
    )
    driver.implicitly_wait(0.5)

    text_box = driver.find_element(by=By.ID, value="selectPage")
    text_box.send_keys("120")

    # Wait the correct number of rows
    while (
        len(
            driver.find_elements(
                By.XPATH,
                '//*[@id="divContainerIframeB3"]/div/div[1]/form/div[2]/div/table/tbody/tr',
            )
        )
        < 21
    ):
        driver.implicitly_wait(5)

    table = (
        WebDriverWait(driver, 20)
        .until(
            EC.visibility_of_element_located(
                (
                    By.XPATH,
                    '//*[@id="divContainerIframeB3"]/div/div[1]/form/div[2]/div/table',
                )
            )
        )
        .get_attribute("outerHTML")
    )
    df1 = pd.read_html(table)[0]

    # Close the browser
    driver.quit()

    df1.drop(df1.tail(2).index, inplace=True)  # drop last 2 rows
    return df1


# Fill missing values based on the dates present in the index list
def fill_missing_values(index_historical, stocks_historical):
    for date in index_historical.index:
        for dict_key in stocks_historical.keys():
            stocks_historical[dict_key] = stocks_historical[dict_key].sort_index()
            if not date in stocks_historical[dict_key].index or np.isnan(
                stocks_historical[dict_key][date]
            ):
                index_match = stocks_historical[dict_key].index.get_indexer(
                    [date], method="nearest"
                )
                stocks_historical[dict_key][date] = stocks_historical[dict_key].iloc[
                    index_match[0]
                ]


# Create train dataset
def create_train_dataset(index_historical, stocks_historical, cutoff_date):
    index_historical_train = index_historical[index_historical.index < cutoff_date]

    stocks_historical_train = {}
    for dict_key in stocks_historical.keys():
        stocks_historical_train[dict_key] = stocks_historical[dict_key][
            stocks_historical[dict_key].index < cutoff_date
        ]

    return index_historical_train, stocks_historical_train


# Create test dataset
def create_test_dataset(index_historical, stocks_historical, cutoff_date):
    index_historical_test = index_historical[index_historical.index >= cutoff_date]

    stocks_historical_test = {}
    for dict_key in stocks_historical.keys():
        stocks_historical_test[dict_key] = stocks_historical[dict_key][
            stocks_historical[dict_key].index >= cutoff_date
        ]

    return index_historical_test, stocks_historical_test


# -------------------------------


# FUNCTIONS FOR GUROBI WITH VALUE LIMIT:
def get_last_quotes(tickers):
    last_quotes = {}
    for ticker in tickers:
        ticker_yahoo = yf.Ticker(ticker)
        data = ticker_yahoo.history()
        last_quote = data["Close"].iloc[-1]
        last_quotes[ticker] = last_quote
    return last_quotes


def normalize_stocks(stocks):
    total_stocks = sum(stocks.values())
    if total_stocks == 0:
        return {stock: 0 for stock in stocks.keys()}
    normalized_stocks = {
        stock: num_stocks / total_stocks for stock, num_stocks in stocks.items()
    }
    return normalized_stocks


def buy_exact_stocks(max_value, weights, prices):
    stocks_to_buy = {}

    for stock, weight in weights.items():
        allocation = max_value * weight
        num_stocks = min(allocation // prices[stock], max_value // prices[stock])
        if num_stocks > 0:
            stocks_to_buy[stock] = num_stocks

    return stocks_to_buy


# ----------------------------------------


def get_all_selected_stocks_accumulated_return_rebalanceamento(
    stocks_historical, weights, index_historical, start_value
):
    def get_stock_return(ticker, stocks_historical):
        temp_return = []
        last_date = None
        for date in index_historical.index:
            if last_date != None:
                temp_return.append(
                    stocks_historical[ticker][date]
                    / stocks_historical[ticker][last_date]
                    - 1
                )
            last_date = date
        return temp_return

    selected_stocks_returns = {}
    for dict_key in weights.keys():
        selected_stocks_returns[dict_key] = get_stock_return(
            dict_key, stocks_historical
        )

    days = list(index_historical.keys())
    tickers = list(weights.keys())

    stock_values = np.zeros((len(tickers), len(days) - 1))

    for i in range(len(days) - 1):  # days
        day_total_value = 0
        for j in range(len(tickers)):  # tickers
            if i == 0:
                stock_values[j][i] = weights[tickers[j]] * start_value
            else:
                stock_values[j][i] = stock_values[j][i - 1] * (
                    1 + selected_stocks_returns[tickers[j]][i]
                )

            day_total_value += stock_values[j][i]

    days.pop(0)
    return pd.Series(np.sum(stock_values, axis=0), index=days)
