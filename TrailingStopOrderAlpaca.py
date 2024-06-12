from alpaca.trading.client import TradingClient
import keyring as kr
from alpaca.trading.requests import TrailingStopOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import json
import requests
import time

APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
APCA_API_KEY_ID = kr.get_password("AlpacaKEY","drcook6611")
APCA_API_SECRET_KEY = kr.get_password("AlpacaSecret","drcook6611")

trading_client = TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
account = trading_client.get_account()

if account.trading_blocked:
    print('Account is currently restricted from trading.')
print('${} is available as buying power.'.format(account.buying_power))

symbol = 'AAL240614P00011500'

limit_order_data = LimitOrderRequest(
    symbol=symbol,
    qty=1,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY,
    limit_price=29,
)

# Limit order
limit_order = trading_client.submit_order(
    order_data=limit_order_data
)

price = limit_order.filled_avg_price
if price == None:
    price = 0.25

while True:
    print('Checking price...')
    url = f"https://data.alpaca.markets/v1beta1/options/quotes/latest?symbols={symbol}&feed=indicative"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": APCA_API_KEY_ID,
        "APCA-API-SECRET-KEY": APCA_API_SECRET_KEY
    }

    response = requests.get(url, headers=headers)

    current_bid = json.loads(response.content.decode('utf-8')).get('quotes').get(symbol).get('bp')

    if ((current_bid - price) / current_bid) >= 0.15:
        break

    time.sleep(5)

limit_order_data = LimitOrderRequest(
    symbol=symbol,
    qty=1,
    side=OrderSide.SELL,
    time_in_force=TimeInForce.DAY,
    limit_price=json.loads(response.content.decode('utf-8')).get('quotes').get(symbol).get('bp'),
)

# Limit order
limit_order = trading_client.submit_order(
    order_data=limit_order_data
)