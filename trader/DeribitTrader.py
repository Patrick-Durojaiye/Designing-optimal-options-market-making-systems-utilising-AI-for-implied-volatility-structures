import json
import requests
from Order import Order


class DeribitTrader:

    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = ""

    def _make_request(self, endpoint: str, method: str, params=None) -> dict:
        """

        :param endpoint:
        :param method:
        :param params:
        :return:
        """
        url = f"{self.base_url}{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        response = requests.request(method, url, headers=headers, json=params)
        return response.json()

    def submit_order(self, method: str, side: str, instrument_name: str, amount: float, type: str,
                     price: float) -> Order:

        if side == "buy":
            endpoint = "/private/buy"

        elif side == "sell":
            endpoint = "/private/sell"

        params = {
            "instrument_name": instrument_name,
            "amount": amount,
            "type": type,
            "price": price
        }

        response = self._make_request(endpoint=endpoint, method=method, params=params)

        if response["result"]:
            order_id = response["result"]["order"]["order_id"]
            order_timestamp = response["result"]["order"]["creation_timestamp"]
            expiry = instrument_name[4:11]
            strike = int(instrument_name[12:-2])
            order = Order(strike=strike, expiry=expiry, side=side, order_id=order_id, order_timestamp=order_timestamp)
            print("Order submission successful")
            return order
        else:
            print("Order submission failed")


