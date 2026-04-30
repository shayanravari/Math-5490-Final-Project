import numpy as np
from numba import njit

@njit
def solve_hjb_offline(lot_size=100.0):
    T = 300.0             
    N_t = 1000            
    dt = T / N_t
    
    Q_bar = 10            
    N_q = 2 * Q_bar + 1   
    
    alpha_max = 5.0
    N_alpha = 21
    d_alpha = (2 * alpha_max) / (N_alpha - 1)
    
    # Financial parameters
    kappa = 0.5           
    phi = 0.005           # Inventory penalty
    psi = 0.001           # Terminal penalty
    upsilon_mo = 0.005    # MO Fee
    sigma = 0.01          # Tick size
    lam_arrival = 2.0     
    
    V_bins = np.array([1.0, 2.0, 5.0])
    V_probs = np.array([0.6, 0.3, 0.1])
    
    q_grid = np.arange(-Q_bar, Q_bar + 1.0)
    alpha_grid = np.linspace(-alpha_max, alpha_max, N_alpha)
    
    V_next = np.zeros((N_alpha, N_q))
    V_curr = np.zeros((N_alpha, N_q))
    
    opt_delta_ask = np.zeros((N_t, N_alpha, N_q), dtype=np.int32)
    opt_z_ask = np.zeros((N_t, N_alpha, N_q), dtype=np.int32)
    opt_delta_bid = np.zeros((N_t, N_alpha, N_q), dtype=np.int32)
    opt_z_bid = np.zeros((N_t, N_alpha, N_q), dtype=np.int32)
    opt_mo_size = np.zeros((N_t, N_alpha, N_q), dtype=np.int32)
    
    for i_a in range(N_alpha):
        for i_q in range(N_q):
            q = q_grid[i_q]
            V_next[i_a, i_q] = -np.abs(q) * lot_size * upsilon_mo - psi * (q**2)
            
    for t_idx in range(N_t - 1, -1, -1):
        for i_a in range(N_alpha):
            for i_q in range(N_q):
                alpha = alpha_grid[i_a]
                q = q_grid[i_q]
                
                drift_term = 0.0
                if i_a > 0 and i_a < N_alpha - 1:
                    drift_coeff = -kappa * alpha
                    if drift_coeff > 0: 
                        drift_term = drift_coeff * (V_next[i_a+1, i_q] - V_next[i_a, i_q]) / d_alpha
                    else:               
                        drift_term = drift_coeff * (V_next[i_a, i_q] - V_next[i_a-1, i_q]) / d_alpha
                
                continuous_pde = drift_term - phi * (q**2) + (q * lot_size * sigma * alpha)
                
                # --- Ask Side (Sell LO) ---
                best_H_ask = -1e9
                best_d_ask = 1
                best_z_ask = 0
                
                for delta in range(1, 4):
                    for z in range(0, int(Q_bar + q) + 1):
                        expected_fill_val = 0.0
                        
                        # Exponentially increasing Queue Depth D
                        if delta == 1:
                            D = max(0.0, 0.5 - 0.2 * alpha)   # 50 shares ahead
                        elif delta == 2:
                            D = max(0.0, 3.0 - 0.2 * alpha)   # 300 shares ahead
                        else:
                            D = max(0.0, 8.0 - 0.2 * alpha)   # 800 shares ahead
                            
                        # If doing nothing (z=0), force Y=0 but still calc alpha jump
                        if z == 0: D = 9999.0 
                        
                        for k in range(len(V_bins)):
                            V_mo = V_bins[k]
                            prob = V_probs[k]
                            Y = min(float(z), max(0.0, V_mo - D))
                            
                            alpha_jump = alpha + 0.1 * V_mo
                            a_j_idx = int(round((alpha_jump + alpha_max) / d_alpha))
                            a_j_idx = min(N_alpha-1, max(0, a_j_idx))
                            q_new_idx = min(N_q-1, max(0, i_q - int(Y)))
                            
                            val_jump = V_next[a_j_idx, q_new_idx] - V_next[i_a, i_q] + (Y * lot_size * delta * sigma)
                            expected_fill_val += prob * val_jump
                            
                        H_ask = lam_arrival * expected_fill_val
                        if H_ask > best_H_ask:
                            best_H_ask = H_ask
                            best_d_ask = delta
                            best_z_ask = z
                            
                # --- Bid Side (Buy LO) ---
                best_H_bid = -1e9
                best_d_bid = 1
                best_z_bid = 0
                
                for delta in range(1, 4):
                    for z in range(0, int(Q_bar - q) + 1):
                        expected_fill_val = 0.0
                        
                        if delta == 1:
                            D = max(0.0, 0.5 + 0.2 * alpha)   
                        elif delta == 2:
                            D = max(0.0, 3.0 + 0.2 * alpha)
                        else:
                            D = max(0.0, 8.0 + 0.2 * alpha)
                            
                        if z == 0: D = 9999.0
                        
                        for k in range(len(V_bins)):
                            V_mo = V_bins[k]
                            prob = V_probs[k]
                            Y = min(float(z), max(0.0, V_mo - D))
                            
                            alpha_jump = alpha - 0.1 * V_mo
                            a_j_idx = int(round((alpha_jump + alpha_max) / d_alpha))
                            a_j_idx = min(N_alpha-1, max(0, a_j_idx))
                            q_new_idx = min(N_q-1, max(0, i_q + int(Y)))
                            
                            val_jump = V_next[a_j_idx, q_new_idx] - V_next[i_a, i_q] + (Y * lot_size * delta * sigma)
                            expected_fill_val += prob * val_jump
                            
                        H_bid = lam_arrival * expected_fill_val
                        if H_bid > best_H_bid:
                            best_H_bid = H_bid
                            best_d_bid = delta
                            best_z_bid = z
                
                continuous_pde += best_H_ask + best_H_bid
                
                # --- Market Order Obstacle ---
                M_val = -1e9
                best_mo_size = 0
                
                if q > 0:
                    for m in range(1, int(q) + 1):
                        val = V_next[i_a, i_q - m] - (m * lot_size * upsilon_mo)
                        if val > M_val:
                            M_val = val
                            best_mo_size = -m 
                            
                if q < 0:
                    for m in range(1, int(-q) + 1):
                        val = V_next[i_a, i_q + m] - (m * lot_size * upsilon_mo)
                        if val > M_val:
                            M_val = val
                            best_mo_size = m 

                # Explicit Projection
                V_unconstrained = V_next[i_a, i_q] + dt * continuous_pde
                
                if M_val > V_unconstrained:
                    V_curr[i_a, i_q] = M_val
                    opt_mo_size[t_idx, i_a, i_q] = best_mo_size
                else:
                    V_curr[i_a, i_q] = V_unconstrained
                    opt_mo_size[t_idx, i_a, i_q] = 0
                
                opt_delta_ask[t_idx, i_a, i_q] = best_d_ask
                opt_z_ask[t_idx, i_a, i_q] = best_z_ask
                opt_delta_bid[t_idx, i_a, i_q] = best_d_bid
                opt_z_bid[t_idx, i_a, i_q] = best_z_bid
                    
        V_next[:] = V_curr[:]
        
    return opt_delta_ask, opt_z_ask, opt_delta_bid, opt_z_bid, opt_mo_size


# ==========================================
# 2. THE AGENT (THE BODY)
# ==========================================

class HJBMarketMaker:
    def __init__(self, action_interval=0.1, lot_size=100):
        self.action_interval = action_interval
        self.lot_size = lot_size
        
        print(f"Solving HJBQVI Offline (Lot Size = {self.lot_size})... Please wait.")
        self.opt_d_ask, self.opt_z_ask, self.opt_d_bid, self.opt_z_bid, self.opt_mo = solve_hjb_offline(float(self.lot_size))
        print("HJBQVI Solved! Tables loaded.")
        
        self.last_action_time = 0.0
        self.T = 300.0
        self.N_t = 1000
        self.alpha_max = 5.0
        self.N_alpha = 21
        self.Q_bar = 10
        self.current_alpha = 0.0
        self.kappa = 0.5
        self.last_tick_time = 0.0
        
        self.active_ask_price = None
        self.active_ask_size = 0
        self.active_bid_price = None
        self.active_bid_size = 0
        self.last_inventory = 0

    def _update_alpha(self, env):
        dt = env.current_time - self.last_tick_time
        self.current_alpha *= np.exp(-self.kappa * dt)
        
        for trade in env.recent_trades:
            if not trade.get('is_agent', False):
                vol_lots = trade['size'] / self.lot_size
                if trade['side'] == 'BUY':
                    self.current_alpha += 0.1 * vol_lots
                else:
                    self.current_alpha -= 0.1 * vol_lots
                    
        self.last_tick_time = env.current_time

    def get_actions(self, env):
        self._update_alpha(env)
        
        if env.current_time - self.last_action_time < self.action_interval:
            return None
            
        self.last_action_time = env.current_time
        actions =[]
        
        if env.inventory != self.last_inventory:
            self.active_ask_price = None
            self.active_bid_price = None
            self.last_inventory = env.inventory

        # --- Hard Cap Emergency Dump ---
        max_allowed_shares = self.Q_bar * self.lot_size
        if env.inventory > max_allowed_shares:
            excess = env.inventory - max_allowed_shares
            actions.extend([{'type': 'cancel', 'side': 'ask'}, {'type': 'cancel', 'side': 'buy'}])
            actions.append({'type': 'market', 'side': 'sell', 'size': excess})
            self.active_ask_price, self.active_bid_price = None, None
            print(f"[{env.current_time:.2f}] HARD CAP: Firing SELL MO for {excess} shares.")
            return actions
        elif env.inventory < -max_allowed_shares:
            excess = abs(env.inventory) - max_allowed_shares
            actions.extend([{'type': 'cancel', 'side': 'ask'}, {'type': 'cancel', 'side': 'buy'}])
            actions.append({'type': 'market', 'side': 'buy', 'size': excess})
            self.active_ask_price, self.active_bid_price = None, None
            print(f"[{env.current_time:.2f}] HARD CAP: Firing BUY MO for {excess} shares.")
            return actions
        
        t_idx = int((env.current_time / self.T) * self.N_t)
        t_idx = min(self.N_t - 1, max(0, t_idx))
        
        d_alpha = (2 * self.alpha_max) / (self.N_alpha - 1)
        a_idx = int(round((self.current_alpha + self.alpha_max) / d_alpha))
        a_idx = min(self.N_alpha - 1, max(0, a_idx))
        
        inv_lots = int(env.inventory / self.lot_size)
        q_idx = min(2 * self.Q_bar, max(0, inv_lots + self.Q_bar))

        # --- Check PDE Market Order (Emergency) ---
        mo_decision = self.opt_mo[t_idx, a_idx, q_idx]
        if mo_decision != 0:
            actions.extend([{'type': 'cancel', 'side': 'ask'}, {'type': 'cancel', 'side': 'buy'}])
            self.active_ask_price, self.active_bid_price = None, None
            
            side = 'buy' if mo_decision > 0 else 'sell'
            size_shares = abs(mo_decision) * self.lot_size
            actions.append({'type': 'market', 'side': side, 'size': size_shares})
            print(f"[{env.current_time:.2f}] PDE EMERGENCY: Firing {side.upper()} MO for {size_shares} shares.")
            return actions

        # --- Limit Orders with Inside-the-Spread Logic ---
        d_ask = self.opt_d_ask[t_idx, a_idx, q_idx]
        z_ask = self.opt_z_ask[t_idx, a_idx, q_idx]
        d_bid = self.opt_d_bid[t_idx, a_idx, q_idx]
        z_bid = self.opt_z_bid[t_idx, a_idx, q_idx]

        tick = env.lob.tick_size
        current_spread = env.lob.spread_in_ticks
        
        # Calculate Ask Target
        if z_ask > 0:
            if current_spread > 1:
                # If spread is wide, step inside! d_ask=1 maps to Ask-1 tick.
                target_ask_price = round(env.lob.p_ask0 + (d_ask - 2) * tick, 4)
                # Ensure we don't accidentally cross the bid
                target_ask_price = max(target_ask_price, round(env.lob.p_bid0 + tick, 4))
            else:
                # Normal 1-tick spread mapping
                target_ask_price = round(env.lob.p_ask0 + (d_ask - 1) * tick, 4)
            target_ask_size = z_ask * self.lot_size
        else:
            target_ask_price, target_ask_size = None, 0

        # Calculate Bid Target
        if z_bid > 0:
            if current_spread > 1:
                # If spread is wide, step inside! d_bid=1 maps to Bid+1 tick.
                target_bid_price = round(env.lob.p_bid0 - (d_bid - 2) * tick, 4)
                # Ensure we don't accidentally cross the ask
                target_bid_price = min(target_bid_price, round(env.lob.p_ask0 - tick, 4))
            else:
                target_bid_price = round(env.lob.p_bid0 - (d_bid - 1) * tick, 4)
            target_bid_size = z_bid * self.lot_size
        else:
            target_bid_price, target_bid_size = None, 0

        # --- Smart Cancel/Replace Execution ---
        if (target_ask_price != self.active_ask_price) or (target_ask_size != self.active_ask_size):
            actions.append({'type': 'cancel', 'side': 'ask'})
            if target_ask_size > 0:
                actions.append({'type': 'limit', 'side': 'sell', 'price': target_ask_price, 'size': target_ask_size})
            self.active_ask_price = target_ask_price
            self.active_ask_size = target_ask_size

        if (target_bid_price != self.active_bid_price) or (target_bid_size != self.active_bid_size):
            actions.append({'type': 'cancel', 'side': 'buy'})
            if target_bid_size > 0:
                actions.append({'type': 'limit', 'side': 'buy', 'price': target_bid_price, 'size': target_bid_size})
            self.active_bid_price = target_bid_price
            self.active_bid_size = target_bid_size

        return actions