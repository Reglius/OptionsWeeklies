import time
from datetime import datetime, timedelta
from math import floor
import keyring as kr
import json
import requests

TRADIER_API_TOKEN = kr.get_password("TradierAPI", "drcook6611")
ACCOUNT_ID = kr.get_password("TradierAcct", "drcook6611")

def wait_until(target_time):
    """Wait until the target time (a datetime object)."""
    while datetime.now() < target_time:
        # Sleep for a short period to avoid busy-waiting
        print('Waiting for 08:30 -', end='\r')
        time.sleep(1)
        print('Waiting for 08:30 |', end='\r')
        time.sleep(1)

def wait_until_order_filled(order_id):
    while True:
        response = requests.get(f"https://api.tradier.com/v1/accounts/{ACCOUNT_ID}/orders/{order_id}",
                                headers={'Authorization': f'Bearer {TRADIER_API_TOKEN}', 'Accept': 'application/json'}
                                )
        order = response.json()['order']
        if order['status'] == 'filled':
            return float(order['avg_fill_price'])
        print('Waiting for order to fill -', end='\r')
        time.sleep(1)
        print('Waiting for order to fill |', end='\r')
        time.sleep(1)

def run_at_specific_time(hour, minute):
    """Run the code at the specific hour and minute."""
    now = datetime.now()
    target_time = datetime(now.year, now.month, now.day, hour, minute)

    wait_until(target_time)

    print("It's 08:30! Running the code...")

    # Get account buying power
    response = requests.get(f'https://api.tradier.com/v1/accounts/{ACCOUNT_ID}/balances',
                            params={},
                            headers={'Authorization': f'Bearer {TRADIER_API_TOKEN}', 'Accept': 'application/json'}
                            )

    buying_power = float(response.json()['balances']['cash']['cash_available'])

    print('${} is available as buying power.'.format(buying_power))

    symbol = 'WMT240621C00066670'

    response = requests.get('https://api.tradier.com/v1/markets/quotes',
                            params={'symbols': symbol, 'greeks': 'false'},
                            headers={'Authorization': f'Bearer {TRADIER_API_TOKEN}', 'Accept': 'application/json'}
                            )

    current_bid = float(response.json()['quotes']['quote']['bid'])
    current_ask = float(response.json()['quotes']['quote']['ask'])

    if (current_bid - current_ask) / current_bid > .07:
        print("Spread is too far apart.")
        exit()

    purchase_price = current_ask
    purchase_quantity = floor(buying_power / purchase_price)

    order_data = {
        "class": "option",
        "symbol": symbol,
        "duration": "day",
        "side": "buy_to_open",
        "quantity": purchase_quantity,
        "type": "limit",
        "price": purchase_price
    }

    response = requests.post(f"https://api.tradier.com/v1/accounts/{ACCOUNT_ID}/orders",
                             headers={'Authorization': f'Bearer {TRADIER_API_TOKEN}', 'Accept': 'application/json'},
                             data=order_data
                             )
    order_id = response.json()['order']['id']

    # Wait until the order is filled and get the filled average price
    filled_avg_price = wait_until_order_filled(order_id)
    highest_price = filled_avg_price

    while True:
        response = requests.get('https://api.tradier.com/v1/markets/quotes',
                                params={'symbols': symbol, 'greeks': 'false'},
                                headers={'Authorization': f'Bearer {TRADIER_API_TOKEN}', 'Accept': 'application/json'}
                                )

        current_bid = float(response.json()['quotes']['quote']['bid'])
        current_ask = float(response.json()['quotes']['quote']['ask'])

        if current_bid is None:
            print("Unable to fetch current bid price.")
            break

        highest_price = max(highest_price, current_bid)

        if ((current_bid - highest_price) / current_bid) <= -0.15:
            order_data = {
                "class": "option",
                "symbol": symbol,
                "duration": "day",
                "side": "sell_to_close",
                "quantity": purchase_quantity,
                "type": "limit",
                "price": current_ask
            }

            response = requests.post(f"https://api.tradier.com/v1/accounts/{ACCOUNT_ID}/orders",
                                     headers={'Authorization': f'Bearer {TRADIER_API_TOKEN}', 'Accept': 'application/json'},
                                     data=order_data
                                     )
            order_id = response.json()['order']['id']
            sold_price = wait_until_order_filled(order_id)
            print(f'Sold {purchase_quantity} {symbol} at {sold_price}')
            exit()

        print('Checking Price -', end='\r')
        time.sleep(1)
        print('Checking Price |', end='\r')
        time.sleep(1)

run_at_specific_time(8, 30)
