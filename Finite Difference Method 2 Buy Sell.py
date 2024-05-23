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
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

def calculate_eod_price(ticker):
    data = yf.download(ticker, start='2022-01-01', progress=False)
    data = data[['Close']].copy()

    # Adding technical indicators
    data.loc[:, 'SMA'] = ta.trend.sma_indicator(data['Close'], window=14)
    data.loc[:, 'EMA'] = ta.trend.ema_indicator(data['Close'], window=14)
    data.loc[:, 'RSI'] = ta.momentum.rsi(data['Close'], window=14)

    # Fill NaN values
    data = data.bfill()  # Use bfill instead of fillna with method

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    lookback = 60
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)

    # Model creation
    model = Sequential()
    model.add(Input(shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=25, batch_size=32, verbose=0)

    num_simulations = 1000
    num_days = 1

    last_60_days = scaled_data[-lookback:]
    X_test = []
    X_test.append(last_60_days)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    simulated_prices = []
    for _ in tqdm(range(num_simulations), desc="Predicting Price", leave=False):
        simulation = []
        X_temp = X_test.copy()

        for _ in range(num_days):
            predicted_price = model.predict(X_temp, verbose=0)
            predicted_price = scaler.inverse_transform(
                np.concatenate((predicted_price, X_temp[0, -1, 1:].reshape(1, -1)), axis=1))[:, 0]

            simulation.append(predicted_price[0])

            new_test_data = np.append(X_temp[0, 1:], scaler.transform(
                np.concatenate((predicted_price.reshape(1, -1), X_temp[0, -1, 1:].reshape(1, -1)), axis=1)), axis=0)
            X_temp = []
            X_temp.append(new_test_data)
            X_temp = np.array(X_temp)
            X_temp = np.reshape(X_temp, (X_temp.shape[0], X_temp.shape[1], X_temp.shape[2]))

        simulated_prices.append(simulation)

    simulated_prices = np.array(simulated_prices)

    average_end_of_week_price = round(np.mean(simulated_prices[:, -1]), 2)

    return average_end_of_week_price

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
                predicted = calculate_eod_price(ticker)
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
                continue
            else:
                break

    today = datetime.now()
    pdf.output(f'D:\\Stocks\\{today.year}\\{today.month}\\{today.day}\\sp500_options_analysis.pdf')

    print('Done')
