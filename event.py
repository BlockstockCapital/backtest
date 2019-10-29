from enum import Enum

EventType = Enum("EventType", "TICK BAR SIGNAL ORDER FILL SENTIMENT")

class Event(object):
    pass


class BarEvent(Event):
    """
    Handles the event of receiving a new market
    open-high-low-close-volume bar, as would be generated
    via common data providers such as Yahoo Finance.
    """
    def __init__(
        self, ticker, time, period,
        open_price, high_price, low_price,
        close_price, volume, adj_close_price=None
    ):
        """
        Initialises the BarEvent.
        Parameters:
        ticker - The ticker symbol, e.g. 'GOOG'.
        time - The timestamp of the bar
        period - The time period covered by the bar in seconds
        open_price - The unadjusted opening price of the bar
        high_price - The unadjusted high price of the bar
        low_price - The unadjusted low price of the bar
        close_price - The unadjusted close price of the bar
        volume - The volume of trading within the bar
        adj_close_price - The vendor adjusted closing price
            (e.g. back-adjustment) of the bar
        Note: It is not advised to use 'open', 'close' instead
        of 'open_price', 'close_price' as 'open' is a reserved
        word in Python.
        """
        self.type = EventType.BAR
        self.ticker = ticker
        self.time = time
        self.period = period
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.volume = volume
        self.adj_close_price = adj_close_price
        self.period_readable = self._readable_period()

    def _readable_period(self):
        """
        Creates a human-readable period from the number
        of seconds specified for 'period'.
        For instance, converts:
        * 1 -> '1sec'
        * 5 -> '5secs'
        * 60 -> '1min'
        * 300 -> '5min'
        If no period is found in the lookup table, the human
        readable period is simply passed through from period,
        in seconds.
        """
        lut = {
            1: "1sec",
            5: "5sec",
            10: "10sec",
            15: "15sec",
            30: "30sec",
            60: "1min",
            300: "5min",
            600: "10min",
            900: "15min",
            1800: "30min",
            3600: "1hr",
            86400: "1day",
            604800: "1wk"
        }
        if self.period in lut:
            return lut[self.period]
        else:
            return "%ssec" % str(self.period)

    def __str__(self):
        format_str = "Type: %s, Ticker: %s, Time: %s, Period: %s, " \
            "Open: %s, High: %s, Low: %s, Close: %s, " \
            "Adj Close: %s, Volume: %s" % (
                str(self.type), str(self.ticker), str(self.time),
                str(self.period_readable), str(self.open_price),
                str(self.high_price), str(self.low_price),
                str(self.close_price), str(self.adj_close_price),
                str(self.volume)
            )
        return format_str

    def __repr__(self):
        return str(self)


class MarketEvent(Event):
    """
       Handles the event of receiving a new market update with
       corresponding bars.
    """
    def __init__(self):
        """
        Initialises the MarketEvent.
        """
        self.type = 'MARKET'


class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.
    This is received by a Portfolio object and acted upon.
    """

    def __init__(self, symbol, datetime, signal_type):
        """
        Initialises the SignalEvent.

        Parameters:
        symbol - The ticker symbol, e.g. 'GOOG'.
        datetime - The timestamp at which the signal was generated.
        signal_type - 'LONG' or 'SHORT'.
        """

        self.type = 'SIGNAL'
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type


class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system.
    The order contains a symbol (e.g. GOOG), a type (market or limit),
    quantity and a direction.
    """

    def __init__(self, symbol, order_type, quantity, direction):
        """
        Initialises the order type, setting whether it is
        a Market order ('MKT') or Limit order ('LMT'), has
        a quantity (integral) and its direction ('BUY' or
        'SELL').

        Parameters:
        symbol - The instrument to trade.
        order_type - 'MKT' or 'LMT' for Market or Limit.
        quantity - Non-negative integer for quantity.
        direction - 'BUY' or 'SELL' for long or short.
        """

        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction

    def print_order(self):
        """
        Outputs the values within the Order.
        """
        print("Order: Symbol={}, Type={}, Quantity={}, Direction={}".
              format(self.symbol, self.order_type, self.quantity, self.direction))


class FillEvent(Event):
    """
    Encapsulates the notion of a Filled Order, as returned
    from a brokerage. Stores the quantity of an instrument
    actually filled and at what price. In addition, stores
    the commission of the trade from the brokerage.
    """

    def __init__(self, timeindex, symbol, exchange, quantity,
                 direction, fill_cost, commission=None):
        """
        Initialises the FillEvent object. Sets the symbol, exchange,
        quantity, direction, cost of fill and an optional
        commission.

        If commission is not provided, the Fill object will
        calculate it based on the trade size and Interactive
        Brokers fees.

        Parameters:
        timeindex - The bar-resolution when the order was filled.
        symbol - The instrument which was filled.
        exchange - The exchange where the order was filled.
        quantity - The filled quantity.
        direction - The direction of fill ('BUY' or 'SELL')
        fill_cost - The holdings value in dollars.
        commission - An optional commission sent from IB.
        """

        self.type = 'FILL'
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost

        # Calculate commission
        if commission is None:
            self.commission = self.calculate_ib_commission()
        else:
            self.commission = commission

    def calculate_ib_commission(self):
        """
        Calculates the fees of trading based on an Interactive
        Brokers fee structure for API, in USD.

        This does not include exchange or ECN fees.

        Based on "US API Directed Orders":
        https://www.interactivebrokers.com/en/index.php?f=commission&p=stocks2
        """
        full_cost = 1.3
        if self.quantity <= 500:
            full_cost = max(1.3, 0.013 * self.quantity)
        else:  # Greater than 500
            full_cost = max(1.3, 0.008 * self.quantity)
        full_cost = min(full_cost, 0.5 / 100.0 * self.quantity * self.fill_cost)
        return full_cost
