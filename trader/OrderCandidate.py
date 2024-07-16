
class OrderCandidate:

    def __init__(self, side: str, instrument_name, amount, type, price):
        """
        Initialises the OrderCandiate class.
        This class is used to store the data of a potential order before it is verified by the proposer

        :param side: Denotes of call or put option, denoted by c or p
        :param instrument_name: Name of option contract
        :param amount: Amount to purchase
        :param type: Denotes type of order (Limit or Maker)
        :param price: Price of option ( can be set in terms of USD or IV )
        """
        self.side = side
        self.instrument_name = instrument_name
        self.amount = amount
        self.type = type
        self.price = price
