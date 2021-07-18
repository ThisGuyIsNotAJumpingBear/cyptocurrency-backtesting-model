"""A module contains various investment strategies"""
from typing import Set


class Transaction:
    """The transaction class.
    This contains:
    1. transaction type. i.e. buy & sell
    2. initial price
    3. transaction value
    3. due date"""
    type: bool  # True for buy, False for sell
    price: float
    value: float
    due: int

    def __init__(self, ttype: bool, tprice: float,
                 tvalue: float, tdue: int = 5) -> None:
        self.type = ttype
        self.price = tprice
        self.value = tvalue
        self.due = tdue

    def next_day(self) -> bool:
        self.due -= 1
        if self.due == 0:
            return False
        return True

    def settlement(self, curr_price: float) -> float:
        value_in_USD = self.price * self.value
        profit = (curr_price - self.price) * self.value
        if profit > 0 and self.type:
            print(True)
        elif profit < 0 and not self.type:
            print(True)
        else:
            print(False)

        print(self.type, profit)

        if self.type:
            return value_in_USD + profit
        return value_in_USD - profit


class Game:
    """game board"""
    asset: float
    transactions: Set[Transaction]
    price: float

    def __init__(self, init_asset: float, curr_price: float) -> None:
        """initialize the game object"""
        self.asset = init_asset
        self.transactions = set()
        self.price = curr_price

    def buy(self) -> None:
        """buy the given item by using 20% of the asset"""
        tvalue = self.asset * 0.2 / self.price
        new = Transaction(True, self.price, tvalue)
        self.transactions.add(new)
        self.asset *= 0.8

    def sell(self) -> None:
        """sell the given item by using 20% of the asset"""
        tvalue = self.asset * 0.2 / self.price
        new = Transaction(False, self.price, tvalue)
        self.transactions.add(new)
        self.asset *= 0.8

    def next_day(self, curr_price: float) -> None:
        """calculate the expired transactions"""
        expired_item = None
        for item in self.transactions:
            expired = not item.next_day()
            if expired:
                expired_item = item
                print(self.asset)
                self.asset += item.settlement(self.price)
                print(self.asset)
                print()
        if expired_item is not None:
            self.transactions.remove(expired_item)
        self.price = curr_price
