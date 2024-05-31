import yfinance as yf
import pandas as pd
import numpy as np
import ta
from fastai.tabular.all import *
import torch
from datetime import datetime, timedelta
from tqdm import tqdm
from fpdf import FPDF
import requests
from bs4 import BeautifulSoup
import re
import os
import traceback


def fetch_data(ticker, period='1y'):
    """Fetch historical stock data for a given ticker and period."""
    stock = yf.Ticker(ticker)
    return stock.history(period=period)


def prepare_data(df):
    """Prepare data for model training by adding target and up columns."""
    df['Next_Close'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    df['Up'] = (df['Next_Close'] > df['Close']).astype(int)
    df.drop(columns=['Next_Close'], inplace=True)
    return df


def create_features(df):
    """Create features for model training."""
    df['Open-Close'] = df['Open'] - df['Close']
    df['High-Low'] = df['High'] - df['Low']
    df['Volume'] = df['Volume']
    return df[['Open-Close', 'High-Low', 'Volume']]


def calculate_technical_indicators(stock_data):
    """Calculate technical indicators for stock data."""
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
    stock_data['Target'] = stock_data['Close'].shift(-1) > stock_data['Close']
    return stock_data.dropna()


def load_model_and_data(ticker):
    """Load model and prepare data for training."""
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
        print("Using GPU for training")
    else:
        print("Using CPU for training")

    learn.fit_one_cycle(5)
    return learn


def predict_single_day(learn, ticker):
    """Predict the stock movement for a single day."""
    predict_df = fetch_data(ticker, period='5d').iloc[[-1]]
    predict_df = create_features(predict_df)

    dl = learn.dls.test_dl(predict_df, with_labels=False)
    predictions, _ = learn.get_preds(dl=dl)
    probabilities = torch.nn.functional.softmax(predictions, dim=1)
    max_probs, preds = torch.max(probabilities, dim=1)

    predicted_direction = "up" if preds[0].item() == 1 else "down"
    confidence = max_probs[0].item()

    return predicted_direction, confidence


def calculate_normal_params(ticker):
    """Calculate mean and standard deviation of daily returns."""
    stock = yf.Ticker(ticker)
    historical_data = stock.history(period='1y')
    daily_changes = historical_data['Close'].pct_change().dropna()

    mean_change = daily_changes.mean()
    std_dev_change = daily_changes.std()

    return mean_change, std_dev_change


def monte_carlo_simulation_stock(ticker, initial_price, num_simulations=1000):
    """Run Monte Carlo simulation for stock price."""
    learn = load_model_and_data(ticker)
    mean_change, std_dev_change = calculate_normal_params(ticker)
    final_prices = []

    for _ in range(num_simulations):
        direction, confidence = predict_single_day(learn, ticker)
        change_percent = np.random.normal(mean_change, std_dev_change)
        if direction == "down":
            change_percent = -change_percent

        new_price = initial_price * (1 + change_percent)
        final_prices.append(new_price)

    return final_prices


def monte_carlo_simulation_option(S, K, T, r, sigma, steps, simulations, option_type):
    """Run Monte Carlo simulation for option pricing."""
    dt = T / steps
    price_paths = np.zeros((steps + 1, simulations))
    price_paths[0] = S

    for t in range(1, steps + 1):
        z = np.random.standard_normal(simulations)
        price_paths[t] = price_paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

    if option_type == 'call':
        payoff = np.maximum(price_paths[-1] - K, 0)
    else:
        payoff = np.maximum(K - price_paths[-1], 0)

    option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price


def calculate_moving_averages(data, short_window, long_window):
    """Calculate short-term and long-term moving averages."""
    data['Short_MA'] = data['Adj Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long_MA'] = data['Adj Close'].rolling(window=long_window, min_periods=1).mean()
    return data


def calculate_buying_decision(data):
    """Determine if it's a good time to buy based on moving averages."""
    latest_short_ma = data['Short_MA'].iloc[-1]
    latest_long_ma = data['Long_MA'].iloc[-1]

    if latest_short_ma > latest_long_ma:
        return 'call'
    else:
        return 'put'


def get_risk_free_rate():
    """Get the current risk-free rate."""
    return 0.045


def fetch_option_data(ticker):
    """Fetch option data for a given ticker."""
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
    """Filter options that expire within the next week."""
    today = pd.Timestamp.today()
    next_week = today + pd.Timedelta(days=7)
    return options[(pd.to_datetime(options['expirationDate']) >= today) & (pd.to_datetime(options['expirationDate']) <= next_week)]


def next_friday():
    """Get the date of the next Friday."""
    today = datetime.today()
    days_until_friday = (4 - today.weekday() + 7) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    return today + timedelta(days=days_until_friday)


def analyze_options(options, S, r, sigma, steps, simulations):
    """Analyze options to determine if they are good buys."""
    results = []

    for index, row in tqdm(options.iterrows(), total=options.shape[0], desc="Analyzing options", leave=False):
        K = row['strike']
        T = (next_friday() - pd.Timestamp.today()).days / 365.0
        option_type = row['type']
        market_price = row['lastPrice']
        simulated_price = round(monte_carlo_simulation_option(S, K, T, r, sigma, steps, simulations, option_type), 2)
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


def buy_option(ticker):
    """Determine if it's a good time to buy an option based on moving averages."""
    data = yf.download(ticker, start='2022-01-01', progress=False)
    data = calculate_moving_averages(data, short_window=10, long_window=50)

    if data['Short_MA'].iloc[-1] > data['Long_MA'].iloc[-1] and data['Short_MA'].iloc[-2] <= data['Long_MA'].iloc[-2]:
        return True
    if data['Short_MA'].iloc[-1] < data['Long_MA'].iloc[-1] and data['Short_MA'].iloc[-2] >= data['Long_MA'].iloc[-2]:
        return True

    return False


def calculate_historical_volatility(ticker):
    """Calculate historical volatility for a given ticker."""
    stock = yf.Ticker(ticker)
    hist = stock.history()
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    return log_returns.std() * np.sqrt(252)


def create_pdf(option_type, good_buys, news, S, ticker, stock_info, pdf):
    """Create a PDF report with the analysis results."""
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"{ticker}: {stock_info.get('shortName')}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Option Type: {option_type}", ln=True, align='C')

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

    output_dir = fr'D:\Stocks\{datetime.now().year}\{datetime.now().month}\{datetime.now().day}'
    os.makedirs(output_dir, exist_ok=True)

    for ticker in tqdm(symbols, total=len(symbols), desc="Analyzing Stock"):
        while True:
            try:
                ticker = ticker.replace('.', '-')
                stock = yf.Ticker(ticker)
                S = stock.history(period='1d')['Close'].iloc[-1]
                r = get_risk_free_rate()
                if not buy_option(ticker):
                    break
                final_prices = monte_carlo_simulation_stock(ticker, S)
                option_type = 'call' if np.mean(final_prices) > stock.history(period='1d')['Close'].iloc[-1] else 'put'
                options = fetch_option_data(ticker)
                weekly_options = filter_weekly_options(options)
                if weekly_options.empty:
                    break
                sigma = calculate_historical_volatility(ticker)
                results = analyze_options(weekly_options, S, r, sigma, steps=252, simulations=1000)
                good_buys = results[results['type'] == option_type]
                good_buys = good_buys[good_buys['goodBuy']]
                create_pdf(option_type, good_buys, stock.news, S, ticker, stock.info, pdf)
            except Exception:
                print(f'Error with {ticker}')
                print(traceback.format_exc())
                continue
            else:
                break

    pdf.output(os.path.join(output_dir, 'sp500_options_analysis_new.pdf'))

    print('Done')
