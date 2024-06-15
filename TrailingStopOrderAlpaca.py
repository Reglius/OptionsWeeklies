import time
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
import keyring as kr
from alpaca.trading.requests import TrailingStopOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import json
import requests

APCA_API_BASE_URL = 'https://api.alpaca.markets'
# APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
APCA_API_KEY_ID = kr.get_password("AlpacaKEYReal", "drcook6611")
# APCA_API_KEY_ID = kr.get_password("AlpacaKEY", "drcook6611")
APCA_API_SECRET_KEY = kr.get_password("AlpacaSecretReal", "drcook6611")
# APCA_API_SECRET_KEY = kr.get_password("AlpacaSecret", "drcook6611")

trading_client = TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
account = trading_client.get_account()

def wait_until(target_time):
    """Wait until the target time (a datetime object)."""
    while datetime.now() < target_time:
        # Sleep for a short period to avoid busy-waiting
        print('Waiting for 08:30 -', end='\r')
        time.sleep(1)
        print('Waiting for 08:30 |', end='\r')
        time.sleep(1)

def run_at_specific_time(hour, minute):
    """Run the code at the specific hour and minute."""
    now = datetime.now()
    target_time = datetime(now.year, now.month, now.day, hour, minute)

    wait_until(target_time)

    print("It's 08:30! Running the code...")

    if account.trading_blocked:
        print('Account is currently restricted from trading.')
        return

    print('${} is available as buying power.'.format(account.buying_power))

    symbol = 'BX240614C00123000'

    url = f"https://data.alpaca.markets/v1beta1/options/quotes/latest?symbols={symbol}&feed=indicative"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": APCA_API_KEY_ID,
        "APCA-API-SECRET-KEY": APCA_API_SECRET_KEY
    }

    response = requests.get(url, headers=headers)
    data = json.loads(response.content.decode('utf-8'))
    current_bid = data.get('quotes', {}).get(symbol, {}).get('bp')
    current_ask = data.get('quotes', {}).get(symbol, {}).get('ap')

    if (current_bid - current_ask) / current_bid > .07:
        exit()

    limit_order_data = LimitOrderRequest(
        symbol=symbol,
        qty=4,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        limit_price=current_ask,
    )

    limit_order = trading_client.submit_order(
        order_data=limit_order_data
    )

    price = limit_order.filled_avg_price
    highest_price = price

    while True:
        url = f"https://data.alpaca.markets/v1beta1/options/quotes/latest?symbols={symbol}&feed=indicative"

        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": APCA_API_KEY_ID,
            "APCA-API-SECRET-KEY": APCA_API_SECRET_KEY
        }

        response = requests.get(url, headers=headers)
        data = json.loads(response.content.decode('utf-8'))
        current_bid = data.get('quotes', {}).get(symbol, {}).get('bp')
        current_ask = data.get('quotes', {}).get(symbol, {}).get('ap')

        if current_bid is None:
            print("Unable to fetch current bid price.")
            break

        highest_price = max(highest_price, current_bid)

        if ((current_bid - highest_price) / current_bid) <= -0.15:
            limit_order_data = LimitOrderRequest(
                symbol=symbol,
                qty=4,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=current_ask,
            )

            limit_order = trading_client.submit_order(
                order_data=limit_order_data
            )
            exit()

        print('Checking Price -', end='\r')
        time.sleep(1)
        print('Checking Price |', end='\r')
        time.sleep(1)

run_at_specific_time(8, 30)