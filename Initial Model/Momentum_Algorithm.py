import numpy as np

class MAgent:
    def __init__(self, lot_size=100, Q_bar=50):
        self.lot_size = lot_size
        self.Q_bar = Q_bar
        
        self.gamma = 0.015
        self.kappa = 0.5
        self.eta = 0.01 
        
        self.alpha = 0.0
        self.last_time = 0.0

    def update_alpha(self, time_sec, trade_side, trade_size):
        dt = max(0.0, time_sec - self.last_time)
        self.alpha *= np.exp(-self.kappa * dt)
        self.last_time = time_sec
        
        vol_lots = trade_size / self.lot_size
        
        impact = self.eta * np.log1p(vol_lots)
        
        if trade_side == 'BUY':
            self.alpha += impact
        elif trade_side == 'SELL':
            self.alpha -= impact

    def get_action(self, time_sec, inventory, time_to_close_sec, current_bid, current_ask):
        dt = max(0.0, time_sec - self.last_time)
        self.alpha *= np.exp(-self.kappa * dt)
        self.last_time = time_sec
        
        inv_lots = inventory / self.lot_size
        mid_price = (current_bid + current_ask) / 2.0

        r_price = mid_price - (self.gamma * inv_lots * 0.01) + (self.alpha * 0.01)
        
        mo_lots = 0
        z_ask_lots = 0
        z_bid_lots = 0
        
        if inv_lots >= self.Q_bar or r_price < current_bid - 0.02:
            mo_lots = -int(max(1, abs(inv_lots) * 0.5))
            return mo_lots, 0, 0
            
        if inv_lots <= -self.Q_bar or r_price > current_ask + 0.02:
            mo_lots = int(max(1, abs(inv_lots) * 0.5))
            return mo_lots, 0, 0

        if current_ask >= r_price and inv_lots > -self.Q_bar:
            z_ask_lots = 3 if current_ask > r_price + 0.005 else 1
            z_ask_lots = min(z_ask_lots, int(self.Q_bar + inv_lots))
            
        if current_bid <= r_price and inv_lots < self.Q_bar:
            z_bid_lots = 3 if current_bid < r_price - 0.005 else 1
            z_bid_lots = min(z_bid_lots, int(self.Q_bar - inv_lots))
            
        return mo_lots, z_ask_lots, z_bid_lots