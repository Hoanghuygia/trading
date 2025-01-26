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
        current_price = self.data[self.current_step][3]  

        if action == 1:  
            max_shares = self.balance // current_price
            transaction_cost = max_shares * current_price * self.transaction_fee

            if self.balance >= max_shares * current_price + transaction_cost:
                self.shares_held += max_shares
                self.balance -= (max_shares * current_price + transaction_cost)

        elif action == 2:  
            if self.shares_held > 0:
                transaction_revenue = self.shares_held * current_price
                transaction_cost = transaction_revenue * self.transaction_fee
                self.balance += (transaction_revenue - transaction_cost)
                self.shares_held = 0

        reward = (self.balance + self.shares_held * current_price - self.initial_balance) / self.initial_balance
        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1

        return self._get_state(), reward, self.done
      
        