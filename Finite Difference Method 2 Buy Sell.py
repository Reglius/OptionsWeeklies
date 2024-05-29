import requests
import numpy as np
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
from pykalman import KalmanFilter
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import ta

num_simulations = 1000
num_steps = 100

def calculate_technical_indicators(stock_data):
    stock_data['RSI'] = ta.momentum.RSIIndicator(stock_data['Close']).rsi()
    stock_data['EMA10'] = ta.trend.EMAIndicator(stock_data['Close'], window=10).ema_indicator()
    stock_data['EMA50'] = ta.trend.EMAIndicator(stock_data['Close'], window=50).ema_indicator()
    bb = ta.volatility.BollingerBands(stock_data['Close'])
    stock_data['Upper_BB'] = bb.bollinger_hband()
    stock_data['Middle_BB'] = bb.bollinger_mavg()
    stock_data['Lower_BB'] = bb.bollinger_lband()
    stock_data['ATR'] = ta.volatility.AverageTrueRange(stock_data['High'], stock_data['Low'],
                                                       stock_data['Close']).average_true_range()
    stock_data['VWAP'] = ta.volume.VolumeWeightedAveragePrice(stock_data['High'], stock_data['Low'],
                                                              stock_data['Close'],
                                                              stock_data['Volume']).volume_weighted_average_price()
    return stock_data

def predict_stock_movement(ticker):
    data = yf.download(ticker, start='2022-01-01', progress=False)

    stock_data = calculate_technical_indicators(data)

    # Step 3: Prepare the data
    stock_data['Open-Close'] = stock_data['Open'] - stock_data['Close']
    stock_data['High-Low'] = stock_data['High'] - stock_data['Low']
    stock_data['Price_Change'] = stock_data['Close'].pct_change()
    stock_data['MA10'] = stock_data['Close'].rolling(window=10).mean()
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['Target'] = stock_data['Close'].shift(-1) > stock_data['Close']
    stock_data = stock_data.dropna()

    # Step 4: Create features and labels
    features = stock_data[
        ['Open-Close', 'High-Low', 'Price_Change', 'MA10', 'MA50', 'RSI', 'EMA10', 'EMA50', 'Upper_BB', 'Middle_BB',
         'Lower_BB', 'ATR', 'VWAP']]
    labels = stock_data['Target']

    # Step 5: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Step 6: Hyperparameter tuning using Grid Search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 6, 8, 10],
        'criterion': ['gini', 'entropy']
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    latest_data = stock_data.iloc[-1]
    next_day_features = [
        [latest_data['Open-Close'], latest_data['High-Low'], latest_data['Price_Change'], latest_data['MA10'],
         latest_data['MA50'], latest_data['RSI'], latest_data['EMA10'], latest_data['EMA50'], latest_data['Upper_BB'],
         latest_data['Middle_BB'], latest_data['Lower_BB'], latest_data['ATR'], latest_data['VWAP']]]

    simulation_results = []
    for _ in tqdm(range(num_simulations), desc='Random Forest Prediction', leave=False):
        model = RandomForestClassifier(n_estimators=best_model.n_estimators,
                                       max_depth=best_model.max_depth,
                                       criterion=best_model.criterion,
                                       random_state=np.random.randint(0, 10000))
        model.fit(X_train, y_train)
        prediction = model.predict(next_day_features)
        simulation_results.append(prediction[0])

    up_probability = np.mean(simulation_results)
    down_probability = 1 - up_probability

    prediction = 'call' if up_probability > down_probability else 'put'
    confidence = max(up_probability, down_probability)

    return prediction

def moving_average_crossover_strategy(stock, short_window, long_window):
    data = stock.history(period='1y')
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
    signals['signal'] = 0.0
    signals['signal'][short_window:] = np.where(
        signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    signals['action'] = np.where(signals['positions'] == 1.0, 'buy', np.where(signals['positions'] == -1.0, 'sell', 'hold'))

    return signals['action'].iloc[-1]

def analyze_recommendations(stock):
    evaluation = 0
    evaluation = evaluation + (stock.get_recommendations().strongBuy.values[0] * 3)
    evaluation = evaluation + (stock.get_recommendations().buy.values[0] * 1)
    evaluation = evaluation + (stock.get_recommendations().sell.values[0] * -1)
    evaluation = evaluation + (stock.get_recommendations().strongSell.values[0] * -3)
    return 'call' if evaluation >= 0 else 'put'

def apply_kalman_filter(data):
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    state_means, _ = kf.filter(data)
    return state_means

def calculate_eod_price(ticker, sp500, us10yr):
    data = yf.download(ticker, start='2022-01-01', progress=False)
    merged_data = data.join(sp500['SP500_Close'], how='left')
    merged_data = merged_data.join(us10yr['US10Yr_Close'], how='left')

    merged_data.fillna(method='ffill', inplace=True)
    merged_data.fillna(method='bfill', inplace=True)

    merged_data['Kalman_Close'] = apply_kalman_filter(merged_data['Close'].values)
    merged_data['Kalman_SP500_Close'] = apply_kalman_filter(merged_data['SP500_Close'].values)
    merged_data['Kalman_US10Yr_Close'] = apply_kalman_filter(merged_data['US10Yr_Close'].values)
    merged_data = merged_data.drop(columns=['Close', 'SP500_Close', 'US10Yr_Close'])

    merged_data['RSI'] = ta.momentum.RSIIndicator(close=merged_data['Kalman_Close']).rsi()
    macd = ta.trend.MACD(close=merged_data['Kalman_Close'])
    merged_data['MACD'] = macd.macd()
    merged_data['MACD_Signal'] = macd.macd_signal()
    merged_data['MACD_Diff'] = macd.macd_diff()
    bollinger = ta.volatility.BollingerBands(close=merged_data['Kalman_Close'])
    merged_data['BB_High'] = bollinger.bollinger_hband()
    merged_data['BB_Low'] = bollinger.bollinger_lband()
    merged_data['BB_Mid'] = bollinger.bollinger_mavg()

    merged_data.fillna(method='ffill', inplace=True)
    merged_data.fillna(method='bfill', inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(merged_data)

    close_prices = merged_data['Kalman_Close'].values.reshape(-1, 1)
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    price_scaler.fit(close_prices)

    def create_sequences(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step)])
            y.append(data[i + time_step, 6])
        return np.array(X), np.array(y)

    time_step = 60
    X, y = create_sequences(scaled_data, time_step)

    inputs = Input(shape=(time_step, X.shape[2]))
    x = LSTM(units=50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x, training=True)
    x = LSTM(units=50, return_sequences=False)(x)
    x = Dropout(0.2)(x, training=True)
    x = Dense(units=25)(x)
    outputs = Dense(units=1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=1, epochs=1, verbose=0)

    last_60_days = scaled_data[-60:]
    X_input = last_60_days.reshape(1, -1)
    X_input = X_input.reshape((1, time_step, X.shape[2]))

    predicted_prices = []
    for _ in tqdm(range(num_simulations), desc="Modeling Prices", leave=False):
        predicted_price = model(X_input, training=True)  # Ensure dropout is active during prediction
        predicted_price = price_scaler.inverse_transform(predicted_price.numpy())
        predicted_prices.append(predicted_price[0, 0])

    predicted_prices = np.array(predicted_prices)

    mean_predicted_price = np.mean(predicted_prices)
    std_predicted_price = np.std(predicted_prices)

    merged_data['Returns'] = merged_data['Kalman_Close'].pct_change()

    mean_return = merged_data['Returns'].mean()
    volatility = merged_data['Returns'].std()
    simulation_results = np.zeros(num_simulations)

    for i in tqdm(range(num_simulations), desc="Geometric Brownian Motion Model", leave=False):
        price = mean_predicted_price
        daily_return = np.random.normal(mean_return, volatility)
        price *= (1 + daily_return)
        simulation_results[i] = price

    mean_simulated_price = round(np.mean(simulation_results), 2)
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
    # data = yf.download("^IRX", period="1d", progress=False)
    # risk_free_rate = data['Close'].iloc[-1] / 100
    return .045

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

def monte_carlo_simulation(S, K, T, r, sigma, steps, simulations, option_type):
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

def next_friday():
    today = datetime.today()
    days_until_friday = (4 - today.weekday() + 7) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    next_friday_date = today + timedelta(days=days_until_friday)
    return next_friday_date

def analyze_options(options, S, r, sigma, steps, simulations):
    results = []
    for index, row in tqdm(options.iterrows(), total=options.shape[0], desc="Analyzing options", leave=False):
        K = row['strike']
        T = (next_friday() - pd.Timestamp.today()).days / 365.0
        option_type = row['type']
        market_price = row['lastPrice']
        simulated_price = round(monte_carlo_simulation(S, K, T, r, sigma, steps, simulations, option_type), 2)
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
    data = yf.download(ticker, start='2022-01-01', progress=False)
    data = calculate_moving_averages(data, short_window=10, long_window=50)
    if data['Short_MA'].iloc[-1] > data['Long_MA'].iloc[-1] and data['Short_MA'].iloc[-2] <= data['Long_MA'].iloc[-2]:
        return True
    if data['Short_MA'].iloc[-1] < data['Long_MA'].iloc[-1] and data['Short_MA'].iloc[-2] >= data['Long_MA'].iloc[-2]:
        return True
    return False

def create_pdf(signal, sentiment, random_forest, good_buys, predicted, news, close, ticker, stock_info, pdf):
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"{ticker}: {stock_info.get('shortName')}, Current Price: ${round(close, 2)}, Predicted Price: ${predicted}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Signal? {signal}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Sentiment? {sentiment}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Random Forest? {random_forest}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Pricing? {option_type}", ln=True, align='C')

    pdf.set_font("Arial", size=10)
    count = 0
    for new in news:
        if count == 5: continue
        count = count + 1
        pdf.cell(200, 10, txt=re.sub(r'[^A-Za-z\s]', '', new.get('title')), ln=True, align='L', link=new.get('link'))

    pdf.cell(200, 10, txt="Good Buy Weekly Options:", ln=True, align='L')
    if not good_buys.empty:
        for index, option in good_buys.iterrows():
            if (option['marketPrice'] <= 10) & (option['marketPrice'] > .05):
                pdf.cell(200, 10, txt=f"{option['name']}: Strike: {option['strike']}, Expiration: {option['expirationDate']}, Market Price: {option['marketPrice']}, Simulated Price: {option['simulatedPrice']}, Diff: {round(((option['simulatedPrice'] - option['marketPrice']) / option['simulatedPrice']) * 100)}", ln=True, align='L')
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
                signal = moving_average_crossover_strategy(stock, short_window=12, long_window=26)
                sentiment = analyze_recommendations(stock)
                random_forest = predict_stock_movement(ticker)
                options = fetch_option_data(ticker)
                weekly_options = filter_weekly_options(options)
                if weekly_options.empty:
                    break
                mean_simulated_price = calculate_eod_price(ticker, sp500, us10yr)
                option_type = 'call' if mean_simulated_price > stock.history(period='1d')['Close'].iloc[-1] else 'put'
                S = stock.history(period='1d')['Close'].iloc[-1]
                r = get_risk_free_rate()
                sigma = calculate_historical_volatility(ticker)
                results = analyze_options(weekly_options, S, r, sigma, num_steps, num_simulations)

                good_buys = results[results['type'] == option_type]
                good_buys = good_buys[good_buys['goodBuy']]

                create_pdf(signal, sentiment, random_forest, good_buys, mean_simulated_price, stock.news, S, ticker, stock.info, pdf)
            except Exception:
                print(f'Error with {ticker}')
                print(traceback.format_exc())
                continue
            else:
                break

    pdf.output(f'D:\\Stocks\\{datetime.now().year}\\{datetime.now().month}\\{datetime.now().day}\\sp500_options_analysis_new.pdf')

    print('Done')
