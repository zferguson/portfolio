class Bank:

    def __init__(self, balance: l=List[int]):
        self.balance = balance
        
    def transfer(self, account1: int, account2: int, money: int) -> bool:
        if account1-1 not in range(0, len(self.balance)):
            return False
        if account2-1 not in range(0, len(self.balance)):
            return False

        bal_1 = self.balance[account1-1]
        bal_2 = self.balance[account2-1]
        if bal_1 >= money:
            self.balance[account1-1] -= money
            self.balance[account2-1] += money
            return True
        else:
            return False

    def deposit(self, account: int, money: int) -> bool:
        if account-1 not in range(0, len(self.balance)):
            return False

        self.balance[account-1] += money
        return True

    def withdraw(self, account: int, money: int) -> bool:
        if account-1 not in range(0, len(self.balance)):
            return False

        bal = self.balance[account-1]
        if self.balance[account - 1] < money:
            return False
        self.balance[account - 1] -= money
        return True