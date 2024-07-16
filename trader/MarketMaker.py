from abc import ABC, abstractmethod
from OrderCandidate import OrderCandidate


class MarketMaker(ABC):

    def __init__(self, order_book):
        self.order_book = order_book

    @abstractmethod
    def on_tick(self):
        pass

    @abstractmethod
    def create_proposal(self, option_type: str, spot_price: float, strike_price: float, r: float, sigma: float, t: float):
        pass

    @abstractmethod
    def adjust_proposal_to_budget(self, proposal: list):
        pass

    @abstractmethod
    def compute_greeks(self):
        pass

    @abstractmethod
    def generate_quote_prices(self, option_type: str, spot_price: float, strike_price: float, r: float, sigma: float, t: float):
        pass

    @abstractmethod
    def cancel_all_orders(self):
        pass

    @abstractmethod
    def cancel_order(self):
        pass

    @abstractmethod
    def create_order(self):
        pass

    @abstractmethod
    def amend_order(self):
        pass

    @abstractmethod
    def place_orders(self, proposal: list):
        pass

    @abstractmethod
    def place_order(self, order: OrderCandidate):
        pass

    @abstractmethod
    def run(self):
        pass
