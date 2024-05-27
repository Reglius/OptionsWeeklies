import requests
import json

# Replace these with your API key and secret
client_id = 'YOUR_CLIENT_ID'
client_secret = 'YOUR_CLIENT_SECRET'

# OAuth2 token endpoint
token_url = 'https://api.tradestation.com/v2/Security/Authorize'

# API endpoints
options_data_url = 'https://api.tradestation.com/v2/marketdata/options/lookup'
order_url = 'https://api.tradestation.com/v2/accounts/ACCOUNT_ID/orders'  # Replace ACCOUNT_ID with your actual account ID


class TradeStationAPI:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None

    def get_access_token(self):
        payload = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': 'ReadAccount Trade'
        }
        response = requests.post(token_url, data=payload)
        if response.status_code == 200:
            self.access_token = response.json()['access_token']
        else:
            print('Failed to get access token:', response.text)

    def fetch_options_market_data(self, symbol):
        headers = {
            'Authorization': f'Bearer {self.access_token}'
        }
        params = {
            'symbol': symbol
        }
        response = requests.get(options_data_url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print('Failed to fetch options market data:', response.text)
            return None

    def execute_options_trade(self, account_id, symbol, quantity, action, option_type, strike_price, expiration_date):
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        payload = {
            "AccountID": account_id,
            "Symbol": symbol,
            "Quantity": quantity,
            "Action": action,
            "OrderType": "Market",
            "TimeInForce": "Day",
            "OptionType": option_type,
            "StrikePrice": strike_price,
            "ExpirationDate": expiration_date
        }
        response = requests.post(order_url.replace('ACCOUNT_ID', account_id), headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            print('Options trade executed successfully:', response.json())
            return response.json()
        else:
            print('Failed to execute options trade:', response.text)
            return None

    def place_options_stop_limit_order(self, account_id, symbol, quantity, stop_price, limit_price, action, option_type,
                                       strike_price, expiration_date):
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        payload = {
            "AccountID": account_id,
            "Symbol": symbol,
            "Quantity": quantity,
            "Action": action,
            "OrderType": "StopLimit",
            "StopPrice": stop_price,
            "LimitPrice": limit_price,
            "TimeInForce": "GTC",
            "OptionType": option_type,
            "StrikePrice": strike_price,
            "ExpirationDate": expiration_date
        }
        response = requests.post(order_url.replace('ACCOUNT_ID', account_id), headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            print('Stop limit order placed successfully:', response.json())
        else:
            print('Failed to place stop limit order:', response.text)

    def place_options_trailing_stop_order(self, account_id, symbol, quantity, trail_amount, action, option_type,
                                          strike_price, expiration_date):
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        payload = {
            "AccountID": account_id,
            "Symbol": symbol,
            "Quantity": quantity,
            "Action": action,
            "OrderType": "TrailingStop",
            "TrailAmount": trail_amount,
            "TimeInForce": "GTC",
            "OptionType": option_type,
            "StrikePrice": strike_price,
            "ExpirationDate": expiration_date
        }
        response = requests.post(order_url.replace('ACCOUNT_ID', account_id), headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            print('Trailing stop order placed successfully:', response.json())
        else:
            print('Failed to place trailing stop order:', response.text)


# Example usage
if __name__ == '__main__':
    api = TradeStationAPI(client_id, client_secret)
    api.get_access_token()

    if api.access_token:
        symbol = 'AAPL'
        options_data = api.fetch_options_market_data(symbol)
        print(json.dumps(options_data, indent=4))

        # Example options trade execution
        account_id = 'YOUR_ACCOUNT_ID'
        quantity = 1
        action = 'BUY'
        option_type = 'Call'  # or 'Put'
        strike_price = 150  # Example strike price
        expiration_date = '2024-06-21'  # Example expiration date

        trade_response = api.execute_options_trade(account_id, symbol, quantity, action, option_type, strike_price,
                                                   expiration_date)
        if trade_response:
            stop_price = 5  # Example stop price for the option
            limit_price = 4.5  # Example limit price for the option
            trail_amount = 0.5  # Example trail amount for the option

            api.place_options_stop_limit_order(account_id, symbol, quantity, stop_price, limit_price, 'SELL',
                                               option_type, strike_price, expiration_date)
            api.place_options_trailing_stop_order(account_id, symbol, quantity, trail_amount, 'SELL', option_type,
                                                  strike_price, expiration_date)
