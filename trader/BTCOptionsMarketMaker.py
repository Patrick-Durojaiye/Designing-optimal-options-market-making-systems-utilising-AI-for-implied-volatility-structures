from abc import ABC
import time

from trader.MarketMaker import MarketMaker
from trader.OrderCandidate import OrderCandidate
from trader.DeribitTrader import DeribitTrader
from trader.PricingModel import BlackScholes


class BTCOptionsMarketMaker(MarketMaker, ABC, DeribitTrader, BlackScholes):

    def __init__(self, order_book, vol_surface, strikes: list, order_refresh_time: int, api_key: str, api_secret: str):
        """

        :param order_book:
        :param vol_surface:
        :param strikes:
        :param order_refresh_time:
        :param api_key:
        :param api_secret:
        """
        MarketMaker.__init__(self, order_book)
        DeribitTrader.__init__(self, api_key, api_secret)
        BlackScholes.__init__(self)
        self.vol_surface = vol_surface
        self.strikes = strikes
        self.create_timestamp = 0
        self.order_refresh_time = order_refresh_time

    def on_tick(self):
        """

        :return:
        """
        if self.create_timestamp <= time.time():
            self.cancel_all_orders()
            # ToDO: Create methodology for gathering contracts across option chain and execute proposals for each
            #  contract
            proposal = self.create_proposal(option_type, spot_price, strike_price, r, sigma, t)
            proposal_adjusted = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.order_refresh_time + time.time()

    def create_proposal(self, option_type: str, spot_price: float, strike_price: float, r: float, sigma: float,
                        t: float):
        """

        :param option_type:
        :param spot_price:
        :param strike_price:
        :param r:
        :param sigma:
        :param t:
        :return:
        """
        quote = self.generate_quote_prices(option_type, spot_price, strike_price, r, sigma, t)
        bid, ask = quote[0], quote[1]
        return [bid, ask]

    def cancel_order(self):
        """

        :return:
        """
        pass

    def generate_quote_prices(self, option_type: str, spot_price: float, strike_price: float, r: float, sigma: float,
                              t: float):
        """

        :param option_type:
        :param spot_price:
        :param strike_price:
        :param r:
        :param sigma:
        :param t:
        :return: A list containing two elements
            - bid (float) : The bid price
            - ask (float) : The ask price
        """

        if option_type == "C":
            option_value = self.call_value(spot_price, strike_price, r, sigma, t)
        elif option_type == "P":
            option_value = self.put_value(spot_price, strike_price, r, sigma, t)
        spread = 0.01
        bid = option_value * (1+spread)
        ask = option_value * (1-spread)
        quote = [bid, ask]
        return quote

    def adjust_proposal_to_budget(self, proposal: list):
        """

        :param proposal:
        :return:
        """
        proposal_adjusted = []
        return proposal_adjusted

    def place_orders(self, proposal: list[OrderCandidate]):
        """

        :param proposal:
        :return:
        """
        for order in proposal:
            self.place_order(order=order)

    def place_order(self, order: OrderCandidate):
        """

        :param order:
        :return:
        """
        self.submit_order(method="POST", side=order.side, instrument_name=order.instrument_name,
                          amount=order.amount, type=order.type, price=order.price)

    def cancel_all_orders(self):

        # Fetches current orders
        pass

    def manage_orders(self):
        """
        Manages the quotes in the orderbook by adding, cancelling or amending them
        :return:
        """

    def run(self):
        """

        :return:
        """
