class Order:

    def __init__(self, side: str, strike: float, expiry: str, instrument_name: str, amount: float, type: str,
                 price: float, order_id: int, order_timestamp: int):
        """
        Initalises the Order class.
        This class stores data of a recent order submission

        :param side: Denotes if call or put
        :param strike: Strike price of option to purchase
        :param expiry: Expiry of the order
        :param instrument_name: Name of option contract
        :param amount: Amount to purchase
        :param type: Denotes type of order (Limit or Maker)
        :param price: Price of option ( can be set in terms of USD or IV )
        :param order_id: Order id of submitted order on deribit
        :param order_timestamp: Timestamp of when order submitted on deribit
        """
        self.side = side
        self.strike = strike
        self.expiry = expiry
        self.instrument_name = instrument_name
        self.amount = amount
        self.type = type
        self.price = price
        self.order_id = order_id
        self.order_timestamp = order_timestamp
