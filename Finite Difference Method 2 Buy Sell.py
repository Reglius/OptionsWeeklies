import requests
import yfinance as yf
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from fpdf import FPDF
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from tqdm import tqdm
import re
import ta
import warnings
import os
import traceback
import matplotlib.pyplot as plt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# def calculate_eod_price(ticker):
#     data = yf.download(ticker, start='2022-01-01', progress=False)
#     data = data[['Close']].copy()
#
#     # Adding technical indicators
#     data.loc[:, 'SMA'] = ta.trend.sma_indicator(data['Close'], window=14)
#     data.loc[:, 'EMA'] = ta.trend.ema_indicator(data['Close'], window=14)
#     data.loc[:, 'RSI'] = ta.momentum.rsi(data['Close'], window=14)
#
#     # Fill NaN values
#     data = data.bfill()  # Use bfill instead of fillna with method
#
#     # Scale the data
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data)
#
#     lookback = 60
#     X, y = [], []
#     for i in range(lookback, len(scaled_data)):
#         X.append(scaled_data[i - lookback:i])
#         y.append(scaled_data[i, 0])
#
#     X, y = np.array(X), np.array(y)
#
#     # Model creation
#     model = Sequential()
#     model.add(Input(shape=(X.shape[1], X.shape[2])))
#     model.add(LSTM(units=50, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=50, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=50))
#     model.add(Dropout(0.2))
#     model.add(Dense(units=1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#
#     model.fit(X, y, epochs=25, batch_size=32, verbose=0)
#
#     num_simulations = 1000
#     num_days = 1
#
#     last_60_days = scaled_data[-lookback:]
#     X_test = []
#     X_test.append(last_60_days)
#     X_test = np.array(X_test)
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
#
#     simulated_prices = []
#     for _ in tqdm(range(num_simulations), desc="Predicting Price", leave=False):
#         simulation = []
#         X_temp = X_test.copy()
#
#         for _ in range(num_days):
#             predicted_price = model.predict(X_temp, verbose=0)
#             predicted_price = scaler.inverse_transform(
#                 np.concatenate((predicted_price, X_temp[0, -1, 1:].reshape(1, -1)), axis=1))[:, 0]
#
#             simulation.append(predicted_price[0])
#
#             new_test_data = np.append(X_temp[0, 1:], scaler.transform(
#                 np.concatenate((predicted_price.reshape(1, -1), X_temp[0, -1, 1:].reshape(1, -1)), axis=1)), axis=0)
#             X_temp = []
#             X_temp.append(new_test_data)
#             X_temp = np.array(X_temp)
#             X_temp = np.reshape(X_temp, (X_temp.shape[0], X_temp.shape[1], X_temp.shape[2]))
#
#         simulated_prices.append(simulation)
#
#     simulated_prices = np.array(simulated_prices)
#
#     average_end_of_week_price = round(np.mean(simulated_prices[:, -1]), 2)
#
#     return average_end_of_week_price

def calculate_eod_price(ticker, sp500, us10yr):
    data = yf.download(ticker, start='2022-01-01', progress=False)
    merged_data = data.join(sp500['SP500_Close'], how='left')
    merged_data = merged_data.join(us10yr['US10Yr_Close'], how='left')

    merged_data.fillna(method='ffill', inplace=True)
    merged_data.fillna(method='bfill', inplace=True)

    merged_data['RSI'] = ta.momentum.RSIIndicator(close=merged_data['Close']).rsi()
    macd = ta.trend.MACD(close=merged_data['Close'])
    merged_data['MACD'] = macd.macd()
    merged_data['MACD_Signal'] = macd.macd_signal()
    merged_data['MACD_Diff'] = macd.macd_diff()
    bollinger = ta.volatility.BollingerBands(close=merged_data['Close'])
    merged_data['BB_High'] = bollinger.bollinger_hband()
    merged_data['BB_Low'] = bollinger.bollinger_lband()
    merged_data['BB_Mid'] = bollinger.bollinger_mavg()

    merged_data.fillna(method='ffill', inplace=True)
    merged_data.fillna(method='bfill', inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(merged_data)

    close_prices = merged_data['Close'].values.reshape(-1, 1)
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    price_scaler.fit(close_prices)

    def create_sequences(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step)])
            y.append(data[i + time_step, 3])  # Assuming the target is the 4th column (Close price)
        return np.array(X), np.array(y)

    time_step = 60
    X, y = create_sequences(scaled_data, time_step)

    inputs = Input(shape=(time_step, X.shape[2]))
    x = LSTM(units=50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(units=50, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(units=25)(x)
    outputs = Dense(units=1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=1, epochs=1)
    last_60_days = scaled_data[-60:]
    X_input = last_60_days.reshape(1, -1)
    X_input = X_input.reshape((1, time_step, X.shape[2]))

    predicted_price = model.predict(X_input, verbose=0)
    predicted_price = price_scaler.inverse_transform(predicted_price)

    merged_data['Returns'] = merged_data['Close'].pct_change()

    mean_return = merged_data['Returns'].mean()
    volatility = merged_data['Returns'].std()

    num_simulations = 1000
    num_days = 1

    simulation_results = np.zeros(num_simulations)

    for i in range(num_simulations):
        price = predicted_price[0][0]
        for _ in range(num_days):
            daily_return = np.random.normal(mean_return, volatility)
            price = price * (1 + daily_return)
        simulation_results[i] = price

    mean_simulated_price = np.mean(simulation_results)
    median_simulated_price = np.median(simulation_results)
    confidence_interval = np.percentile(simulation_results, [2.5, 97.5])

    return mean_simulated_price


def calculate_moving_averages(data, short_window, long_window):
    data['Short_MA'] = data['Adj Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long_MA'] = data['Adj Close'].rolling(window=long_window, min_periods=1).mean()
    return data

def calculate_buying_decision(data):
    latest_short_ma = data['Short_MA'].iloc[-1]
    latest_long_ma = data['Long_MA'].iloc[-1]

    if latest_short_ma > latest_long_ma:
        return 'call'
    else:
        return 'put'

def get_risk_free_rate():
    data = yf.download("^IRX", period="1d", progress=False)
    risk_free_rate = data['Close'].iloc[-1] / 100
    return risk_free_rate

def fetch_option_data(ticker):
    stock = yf.Ticker(ticker)
    options_dates = stock.options
    option_data = []

    for date in options_dates:
        opt = stock.option_chain(date)
        calls = opt.calls
        puts = opt.puts
        calls['type'] = 'call'
        puts['type'] = 'put'
        calls['expirationDate'] = date
        puts['expirationDate'] = date
        option_data.append(calls)
        option_data.append(puts)

    options = pd.concat(option_data, axis=0)
    return options

def filter_weekly_options(options):
    today = pd.Timestamp.today()
    next_week = today + pd.Timedelta(days=7)
    weekly_options = options[(pd.to_datetime(options['expirationDate']) >= today) &
                             (pd.to_datetime(options['expirationDate']) <= next_week)]
    return weekly_options

def calculate_historical_volatility(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    volatility = log_returns.std() * np.sqrt(252)
    return volatility

def monte_carlo_simulation(S, K, T, r, sigma, steps, simulations, option_type, pred_price):
    dt = T / steps
    price_paths = np.zeros((steps + 1, simulations))
    price_paths[0] = S

    for t in range(1, steps + 1):
        z = np.random.standard_normal(simulations)
        price_paths[t] = price_paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

    price_paths[-1] = pred_price  # Use the predicted end-of-week price for the last step

    if option_type == 'call':
        payoff = np.maximum(price_paths[-1] - K, 0)
    else:
        payoff = np.maximum(K - price_paths[-1], 0)

    option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price

def next_friday():
    today = datetime.today()
    days_until_friday = (4 - today.weekday() + 7) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    next_friday_date = today + timedelta(days=days_until_friday)
    return next_friday_date

def analyze_options(options, S, r, sigma, steps, simulations, pred_price):
    results = []
    for index, row in tqdm(options.iterrows(), total=options.shape[0], desc="Analyzing options", leave=False):
        K = row['strike']
        T = (next_friday() - pd.Timestamp.today()).days / 365.0
        option_type = row['type']
        market_price = row['lastPrice']
        simulated_price = round(monte_carlo_simulation(S, K, T, r, sigma, steps, simulations, option_type, pred_price), 2)
        good_buy = simulated_price > market_price
        results.append({
            'type': option_type,
            'strike': K,
            'expirationDate': row['expirationDate'],
            'marketPrice': market_price,
            'simulatedPrice': simulated_price,
            'goodBuy': good_buy
        })
    return pd.DataFrame(results)

def buy_option(ticker):
    data = yf.download(ticker, start='2022-01-01', progress=False)
    data = calculate_moving_averages(data, short_window=10, long_window=50)
    if data['Short_MA'].iloc[-1] > data['Long_MA'].iloc[-1] and data['Short_MA'].iloc[-2] <= data['Long_MA'].iloc[-2]:
        return True
    if data['Short_MA'].iloc[-1] < data['Long_MA'].iloc[-1] and data['Short_MA'].iloc[-2] >= data['Long_MA'].iloc[-2]:
        return True
    return False

def create_pdf(good_buys, predicted, news, close, ticker, stock_info, pdf):
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"{ticker}: {stock_info.get('shortName')}, Current Price: ${round(close, 2)}, Predicted Price: ${predicted}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Call or Put? {option_type}", ln=True, align='C')

    pdf.set_font("Arial", size=10)
    count = 0
    for new in news:
        if count == 5: continue
        count = count + 1
        pdf.cell(200, 10, txt=re.sub(r'[^A-Za-z\s]', '', new.get('title')), ln=True, align='L')

    pdf.cell(200, 10, txt="Good Buy Weekly Options:", ln=True, align='L')
    if not good_buys.empty:
        for index, option in good_buys.iterrows():
            if option['marketPrice'] <= 10:
                pdf.cell(200, 10, txt=f"Strike: {option['strike']}, Expiration: {option['expirationDate']}, Market Price: {option['marketPrice']}, Simulated Price: {option['simulatedPrice']}, Diff: {round(((option['simulatedPrice'] - option['marketPrice']) / option['simulatedPrice']) * 100)}", ln=True, align='L')
    else:
        pdf.cell(200, 10, txt="No good buy options found.", ln=True, align='L')

    pdf.cell(200, 10, ln=True)

if __name__ == "__main__":
    sp500 = yf.download('^GSPC', start='2022-01-01', progress=False)
    us10yr = yf.download('^TNX', start='2022-01-01', progress=False)
    sp500 = sp500.rename(columns={'Close': 'SP500_Close'})
    us10yr = us10yr.rename(columns={'Close': 'US10Yr_Close'})

    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(str(table))[0]
    symbols = sorted(df['Symbol'].tolist())

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    if not os.path.exists(fr'D:\Stocks\{datetime.now().year}\{datetime.now().month}\{datetime.now().day}'):
        os.makedirs(fr'D:\Stocks\{datetime.now().year}\{datetime.now().month}\{datetime.now().day}')

    for ticker in tqdm(symbols, total=len(symbols), desc="Analyzing Stock"):
        while True:
            try:
                ticker = ticker.replace('.', '-')
                stock = yf.Ticker(ticker)
                if not (buy_option(ticker)):
                    break
                options = fetch_option_data(ticker)
                weekly_options = filter_weekly_options(options)
                if weekly_options.empty:
                    break
                predicted = calculate_eod_price(ticker, sp500, us10yr)
                option_type = 'call' if predicted > stock.history(period='1d')['Close'].iloc[-1] else 'put'
                S = stock.history(period='1d')['Close'].iloc[-1]
                r = get_risk_free_rate()
                sigma = calculate_historical_volatility(ticker)
                results = analyze_options(weekly_options, S, r, sigma, 1000, 10000, predicted)

                good_buys = results[results['type'] == option_type]
                good_buys = good_buys[good_buys['goodBuy']]

                create_pdf(good_buys, predicted, stock.news, S, ticker, stock.info, pdf)
            except Exception:
                print(f'Error with {ticker}')
                print(traceback.format_exc())
                continue
            else:
                break

    today = datetime.now()
    pdf.output(f'D:\\Stocks\\{today.year}\\{today.month}\\{today.day}\\sp500_options_analysis_new.pdf')

    print('Done')
