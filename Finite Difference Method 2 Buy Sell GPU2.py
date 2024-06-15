import traceback
import os
import re
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from bs4 import BeautifulSoup
from fastai.tabular.all import *
import torch
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from arch import arch_model
from fpdf import FPDF
from tqdm import tqdm
import logging
import sys
import contextlib
from io import StringIO

num_simulations = 10000
steps=252

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def fetch_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    return stock.history(period=period)

def prepare_data(df):
    df['Next_Close'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    df['Up'] = (df['Next_Close'] > df['Close']).astype(int)
    df.drop(columns=['Next_Close'], inplace=True)
    return df

def create_features(df):
    df['Open-Close'] = df['Open'] - df['Close']
    df['High-Low'] = df['High'] - df['Low']
    return df[['Open-Close', 'High-Low', 'Volume']]

def calculate_technical_indicators(stock_data):
    stock_data['RSI'] = ta.momentum.RSIIndicator(stock_data['Close']).rsi()
    stock_data['EMA5'] = ta.trend.EMAIndicator(stock_data['Close'], window=5).ema_indicator()
    bb = ta.volatility.BollingerBands(stock_data['Close'])
    stock_data['Upper_BB'] = bb.bollinger_hband()
    stock_data['Middle_BB'] = bb.bollinger_mavg()
    stock_data['Lower_BB'] = bb.bollinger_lband()
    stock_data['ATR'] = ta.volatility.AverageTrueRange(stock_data['High'], stock_data['Low'], stock_data['Close']).average_true_range()
    stock_data['VWAP'] = ta.volume.VolumeWeightedAveragePrice(stock_data['High'], stock_data['Low'], stock_data['Close'], stock_data['Volume']).volume_weighted_average_price()
    stock_data['Open-Close'] = stock_data['Open'] - stock_data['Close']
    stock_data['High-Low'] = stock_data['High'] - stock_data['Low']
    stock_data['Price_Change'] = stock_data['Close'].pct_change()
    stock_data['MA5'] = stock_data['Close'].rolling(window=5).mean()
    stock_data['Target'] = (stock_data['Close'].shift(-1) > stock_data['Close']).astype(int)
    return stock_data.dropna()

def load_model_and_data(ticker):
    df = fetch_data(ticker)
    df = prepare_data(df)
    df = calculate_technical_indicators(df)
    features = create_features(df)
    labels = df['Up']
    combined_df = pd.concat([features, labels], axis=1)

    dls = TabularDataLoaders.from_df(combined_df, path='.', procs=[Categorify, FillMissing, Normalize], y_names="Up", splits=None)
    learn = tabular_learner(dls, metrics=accuracy, layers=[50, 20])

    if torch.cuda.is_available():
        learn.to_fp16()
        # print("Using GPU for training")
    # else:
    #     print("Using CPU for training")

    with suppress_stdout():
        learn.fit_one_cycle(5)

    return learn

def predict_single_day(learn, ticker):
    predict_df = fetch_data(ticker, period='1mo')
    predict_df = create_features(predict_df)
    predict_df = predict_df.iloc[[-1]]

    dl = learn.dls.test_dl(predict_df, with_labels=False)
    predictions, _ = learn.get_preds(dl=dl)
    probabilities = torch.nn.functional.softmax(predictions, dim=1)
    max_probs, preds = torch.max(probabilities, dim=1)

    predicted_direction = "up" if preds[0].item() == 1 else "down"
    confidence = max_probs[0].item()

    return predicted_direction, confidence

def monte_carlo_simulation_option(S, K, T, r, sigma, option_type):
    dt = T / steps
    price_paths = np.zeros((steps + 1, num_simulations))
    price_paths[0] = S

    for t in tqdm(range(1, steps + 1), desc="Geometric Brownian Motion Formula", leave=False):
        z = np.random.standard_normal(num_simulations)
        price_paths[t] = price_paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

    if option_type == 'call':
        payoff = np.maximum(price_paths[-1] - K, 0)
    else:
        payoff = np.maximum(K - price_paths[-1], 0)

    option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price

def get_risk_free_rate():
    return 0.045

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

    return pd.concat(option_data, axis=0)

def filter_weekly_options(options):
    today = pd.Timestamp.today()
    next_week = today + pd.Timedelta(days=7)
    return options[(pd.to_datetime(options['expirationDate']) >= today) & (pd.to_datetime(options['expirationDate']) <= next_week)]

def next_friday():
    today = datetime.today()
    days_until_friday = (4 - today.weekday() + 7) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    return today + timedelta(days=days_until_friday)

def analyze_options(options, S, r, sigma):
    results = []

    for index, row in tqdm(options.iterrows(), total=options.shape[0], desc="Analyzing options", leave=False):
        K = row['strike']
        T = (next_friday() - pd.Timestamp.today()).days / 365.0
        option_type = row['type']
        market_price = row['lastPrice']
        simulated_price = round(monte_carlo_simulation_option(S, K, T, r, sigma, option_type), 2)
        good_buy = simulated_price > market_price
        results.append({
            'name': row['contractSymbol'],
            'type': option_type,
            'strike': K,
            'expirationDate': row['expirationDate'],
            'marketPrice': market_price,
            'simulatedPrice': simulated_price,
            'goodBuy': good_buy
        })

    return pd.DataFrame(results)

def calculate_moving_averages(data, windows):
    for window in windows:
        data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
    return data

def buy_option(ticker):
    data = yf.download(ticker, start='2022-01-01', progress=False)
    data = calculate_moving_averages(data, windows=[5, 8, 13])

    short_ma = data['SMA_5']
    mid_ma = data['SMA_8']
    long_ma = data['SMA_13']

    if short_ma.iloc[-1] > mid_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] <= mid_ma.iloc[-2]:
        return True, 'call'
    if short_ma.iloc[-1] < mid_ma.iloc[-1] < long_ma.iloc[-1] and short_ma.iloc[-2] >= mid_ma.iloc[-2]:
        return True, 'put'

    return False, 'nothing'

def calculate_historical_volatility(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history()
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    return log_returns.std() * np.sqrt(252)

def calcuate_price(data):
    returns = data['Adj Close'].pct_change().dropna()
    returns_rescaled = returns * 100
    model = arch_model(returns_rescaled, vol='Garch', p=1, q=1)
    garch_fit = model.fit(disp='off')

    alpha0 = garch_fit.params['omega']
    alpha1 = garch_fit.params['alpha[1]']
    beta1 = garch_fit.params['beta[1]']
    mu = garch_fit.params['mu'] if 'mu' in garch_fit.params else 0

    learn = load_model_and_data(ticker)
    predicted_direction, confidence = predict_single_day(learn, ticker)

    current_price = data['Adj Close'].iloc[-1]

    num_days = 1

    random_innovations = np.random.normal(size=(num_simulations, num_days))
    simulated_returns = np.zeros((num_simulations, num_days))
    simulated_prices = np.zeros((num_simulations, num_days + 1))
    simulated_prices[:, 0] = current_price

    confidence_factor = 0.01

    for i in tqdm(range(num_simulations), desc="Monte Carlo Price", leave=False):
        sigma_t = np.std(returns)
        for t in range(num_days):
            epsilon_t = random_innovations[i, t]
            sigma_t = np.sqrt(alpha0 + alpha1 * epsilon_t ** 2 + beta1 * sigma_t ** 2)
            adjusted_return = mu + confidence * confidence_factor if predicted_direction == "up" else mu - confidence * confidence_factor
            simulated_returns[i, t] = adjusted_return + sigma_t * epsilon_t
            simulated_prices[i, t + 1] = simulated_prices[i, t] * (1 + simulated_returns[i, t])

    mean_simulated_price = np.mean(simulated_prices[:, -1])
    return mean_simulated_price

def create_pdf(option_type_mc, option_type_ma, good_buys, news, S, ticker, stock_info, pdf):
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"{ticker}: {stock_info.get('shortName')}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Option Type Monte Carlo: {option_type_mc}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Option Type Moving Average: {option_type_ma}", ln=True, align='C')

    pdf.set_font("Arial", size=10)
    for count, new in enumerate(news[:5]):
        pdf.cell(200, 10, txt=re.sub(r'[^A-Za-z\s]', '', new.get('title')), ln=True, align='L', link=new.get('link'))

    pdf.cell(200, 10, txt="Good Buy Weekly Options:", ln=True, align='L')
    if not good_buys.empty:
        for index, option in good_buys.iterrows():
            if 0.05 < option['marketPrice'] <= 10:
                diff = round(((option['simulatedPrice'] - option['marketPrice']) / option['simulatedPrice']) * 100)
                pdf.cell(200, 10, txt=f"{option['name']}: Strike: {option['strike']}, Expiration: {option['expirationDate']}, Market Price: {option['marketPrice']}, Simulated Price: {option['simulatedPrice']}, Diff: {diff}%", ln=True, align='L')
    else:
        pdf.cell(200, 10, txt="No good buy options found.", ln=True, align='L')

    pdf.cell(200, 10, ln=True)

if __name__ == "__main__":
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(StringIO(str(table)))[0]
    symbols = sorted(df['Symbol'].tolist())

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    output_dir = fr'D:\Stocks\{datetime.now().year}\{datetime.now().month}\{datetime.now().day}'
    os.makedirs(output_dir, exist_ok=True)

    for ticker in tqdm(symbols, total=len(symbols), desc="Analyzing Stock"):
        retry_counter = 0
        while True:
            try:
                ticker = ticker.replace('.', '-')
                stock = yf.Ticker(ticker)
                S = stock.history(period='1d')['Close'].iloc[-1]
                r = get_risk_free_rate()
                ma_decision, option_type_ma = buy_option(ticker)
                if not ma_decision:
                    break
                data = yf.download(ticker, start='2022-01-01', progress=False)
                final_prices = calcuate_price(data)
                option_type_mc = 'call' if final_prices > stock.history(period='1d')['Close'].iloc[-1] else 'put'
                options = fetch_option_data(ticker)
                weekly_options = filter_weekly_options(options)
                if weekly_options.empty:
                    break
                sigma = calculate_historical_volatility(ticker)
                results = analyze_options(weekly_options, S, r, sigma)
                good_buys = results[results['type'] == option_type_ma]
                good_buys = good_buys[good_buys['goodBuy']]
                create_pdf(option_type_mc, option_type_ma, good_buys, stock.news, S, ticker, stock.info, pdf)
            except Exception:
                print(f'Error with {ticker}')
                print(traceback.format_exc())
                retry_counter = retry_counter + 1
                print(f'Retry Counter: {retry_counter}')
                if retry_counter == 5:
                    break
                continue
            else:
                break

    pdf.output(os.path.join(output_dir, 'sp500_options_analysis_new.pdf'))

    print('Done')
