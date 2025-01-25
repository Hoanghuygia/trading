class TradingEvironment:
    def __init__(self, data, initial_balance= 10000, transaction_fee= 0.001):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reset()
        
    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        return self.data[self.current_step]
    
    def step(self, action):
        current_prize = self.data[self.current_step][3]
        if action == 1 and self.balance > current_prize:
            max_shares = self.balance / current_prize
            self.shares_held += max_shares
            transaction_cost = max_shares * current_prize * self.transaction_fee
            self.balance -= (max_shares * current_prize + transaction_cost)
        elif action == 2 and self.shares_held > 0:
            self.balance += self.shares_held * current_prize
            transaction_cost = self.shares_held * current_prize * self.transaction_fee
            self.balance -= transaction_cost
            self.shares_held = 0
        
        reward = (self.balance + self.shares_held * current_prize - self.initial_balance) / self.initial_balance   
        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1
        
        return self._get_state(), reward, self.done         
        