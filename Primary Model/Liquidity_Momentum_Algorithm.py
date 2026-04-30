import os
import json
import numpy as np
from numba import njit

@njit
def interp_2d(V_array, a_val, l_val, q_idx, a_max, d_a, N_a, l_min, d_l, N_l):
    # Alpha axis
    f_a = (a_val + a_max) / d_a
    f_a = max(0.0, min(N_a - 1.0, f_a))
    ia0 = int(f_a)
    ia1 = min(ia0 + 1, N_a - 1)
    wa1 = f_a - ia0
    wa0 = 1.0 - wa1
    
    # Lambda axis
    f_l = (l_val - l_min) / d_l
    f_l = max(0.0, min(N_l - 1.0, f_l))
    il0 = int(f_l)
    il1 = min(il0 + 1, N_l - 1)
    wl1 = f_l - il0
    wl0 = 1.0 - wl1
    
    # Extract 4 corner points
    v00 = V_array[ia0, il0, q_idx]
    v01 = V_array[ia0, il1, q_idx]
    v10 = V_array[ia1, il0, q_idx]
    v11 = V_array[ia1, il1, q_idx]
    
    # Interpolate Lambda first, then Alpha
    val0 = wl0 * v00 + wl1 * v01
    val1 = wl0 * v10 + wl1 * v11
    
    return wa0 * val0 + wa1 * val1

# Offline 4D HJB PDE Solver
@njit
def solve_hjb_4d(kappa, eta, gamma, lam_arrival, lambda_bar, lot_size=100.0, Q_bar=50, T=300.0, N_t=1000):
    dt = T / N_t
    N_q = 2 * Q_bar + 1   
    
    # Alpha Grid
    alpha_max = 5.0
    N_alpha = 21
    d_alpha = (2 * alpha_max) / (N_alpha - 1)
    
    # Lambda Grid
    lam_min = 1.0
    lam_max = 20.0
    N_lam = 10
    d_lam = (lam_max - lam_min) / (N_lam - 1)
    
    phi = 5e-6                       
    psi = 0.001                     
    sigma = 0.01                     
    half_spread = 0.005              
    upsilon_mo = 0.0001 
    
    MAX_CLIP_LOTS = 5 
    
    V_bins = np.array([1.0, 3.0, 10.0])
    V_probs = np.array([0.7, 0.2, 0.1])
    
    q_grid = np.arange(-Q_bar, Q_bar + 1.0)
    alpha_grid = np.linspace(-alpha_max, alpha_max, N_alpha)
    lam_grid = np.linspace(lam_min, lam_max, N_lam)
    
    V_next = np.zeros((N_alpha, N_lam, N_q))
    V_curr = np.zeros((N_alpha, N_lam, N_q))
    
    opt_z_ask = np.zeros((N_t, N_alpha, N_lam, N_q), dtype=np.int32)
    opt_z_bid = np.zeros((N_t, N_alpha, N_lam, N_q), dtype=np.int32)
    opt_mo_size = np.zeros((N_t, N_alpha, N_lam, N_q), dtype=np.int32)
    
    # Terminal Condition
    for i_a in range(N_alpha):
        for i_l in range(N_lam):
            for i_q in range(N_q):
                q = q_grid[i_q]
                V_next[i_a, i_l, i_q] = -np.abs(q) * lot_size * upsilon_mo - psi * (q**2)
            
    # Backward Time Stepping
    for t_idx in range(N_t - 1, -1, -1):
        for i_a in range(N_alpha):
            for i_l in range(N_lam):
                for i_q in range(N_q):
                    alpha = alpha_grid[i_a]
                    lam = lam_grid[i_l]
                    q = q_grid[i_q]
                    
                    # Drift Terms
                    drift_alpha = 0.0
                    if 0 < i_a < N_alpha - 1:
                        drift_coeff = -kappa * alpha
                        if drift_coeff > 0: 
                            drift_alpha = drift_coeff * (V_next[i_a+1, i_l, i_q] - V_next[i_a, i_l, i_q]) / d_alpha
                        else:               
                            drift_alpha = drift_coeff * (V_next[i_a, i_l, i_q] - V_next[i_a-1, i_l, i_q]) / d_alpha
                            
                    drift_lam = 0.0
                    if 0 < i_l < N_lam - 1:
                        lam_coeff = 0.5 * (lambda_bar - lam) # Beta = 0.5 resilience
                        if lam_coeff > 0:
                            drift_lam = lam_coeff * (V_next[i_a, i_l+1, i_q] - V_next[i_a, i_l, i_q]) / d_lam
                        else:
                            drift_lam = lam_coeff * (V_next[i_a, i_l, i_q] - V_next[i_a, i_l-1, i_q]) / d_lam
                    
                    continuous_pde = drift_alpha + drift_lam - phi * (q**2) + (q * lot_size * sigma * alpha * 0.1)
                    
                    # Ask Side
                    best_H_ask = -1e9
                    best_z_ask = 0
                    max_sell_lots = min(MAX_CLIP_LOTS, int(Q_bar + q))
                    
                    for z in range(0, max_sell_lots + 1):
                        expected_fill_val = 0.0
                        D = max(0.0, lam * 0.2) if z > 0 else 9999.0 
                        
                        for k in range(len(V_bins)):
                            V_mo = V_bins[k]
                            prob = V_probs[k]
                            Y = min(float(z), max(0.0, V_mo - D)) if z > 0 else 0.0
                            
                            alpha_jump = alpha + eta * (V_mo / (lam ** gamma))
                            lam_jump = max(lam_min, lam - V_mo)
                            q_new_idx = min(N_q-1, max(0, i_q - int(Y)))
                            
                            V_jumped = interp_2d(V_next, alpha_jump, lam_jump, q_new_idx, alpha_max, d_alpha, N_alpha, lam_min, d_lam, N_lam)
                            val_jump = V_jumped - V_next[i_a, i_l, i_q] + (Y * lot_size * half_spread)
                            expected_fill_val += prob * val_jump
                            
                        H_ask = lam_arrival * expected_fill_val
                        if H_ask > best_H_ask:
                            best_H_ask = H_ask
                            best_z_ask = z
                                
                    # Bid Side
                    best_H_bid = -1e9
                    best_z_bid = 0
                    max_buy_lots = min(MAX_CLIP_LOTS, int(Q_bar - q))
                    
                    for z in range(0, max_buy_lots + 1):
                        expected_fill_val = 0.0
                        D = max(0.0, lam * 0.2) if z > 0 else 9999.0
                        
                        for k in range(len(V_bins)):
                            V_mo = V_bins[k]
                            prob = V_probs[k]
                            Y = min(float(z), max(0.0, V_mo - D)) if z > 0 else 0.0
                            
                            alpha_jump = alpha - eta * (V_mo / (lam ** gamma))
                            lam_jump = max(lam_min, lam - V_mo)
                            q_new_idx = min(N_q-1, max(0, i_q + int(Y)))
                            
                            V_jumped = interp_2d(V_next, alpha_jump, lam_jump, q_new_idx, alpha_max, d_alpha, N_alpha, lam_min, d_lam, N_lam)
                            val_jump = V_jumped - V_next[i_a, i_l, i_q] + (Y * lot_size * half_spread)
                            expected_fill_val += prob * val_jump
                            
                        H_bid = lam_arrival * expected_fill_val
                        if H_bid > best_H_bid:
                            best_H_bid = H_bid
                            best_z_bid = z
                    
                    continuous_pde += best_H_ask + best_H_bid
                    
                    # Impulse Control
                    M_val = -1e9
                    best_mo_size = 0
                    if q > 0:
                        for m in range(1, int(q) + 1):
                            val = V_next[i_a, i_l, i_q - m] - (m * lot_size * upsilon_mo)
                            if val > M_val: M_val, best_mo_size = val, -m 
                    if q < 0:
                        for m in range(1, int(-q) + 1):
                            val = V_next[i_a, i_l, i_q + m] - (m * lot_size * upsilon_mo)
                            if val > M_val: M_val, best_mo_size = val, m 

                    V_unconstrained = V_next[i_a, i_l, i_q] + dt * continuous_pde
                    if M_val > V_unconstrained:
                        V_curr[i_a, i_l, i_q] = M_val
                        opt_mo_size[t_idx, i_a, i_l, i_q] = best_mo_size
                    else:
                        V_curr[i_a, i_l, i_q] = V_unconstrained
                        opt_mo_size[t_idx, i_a, i_l, i_q] = 0
                    
                    opt_z_ask[t_idx, i_a, i_l, i_q] = best_z_ask
                    opt_z_bid[t_idx, i_a, i_l, i_q] = best_z_bid
                        
        V_next[:] = V_curr[:]
        
    return opt_z_ask, opt_z_bid, opt_mo_size

class LMAgent:
    def __init__(self, lot_size=100, Q_bar=50, T_horizon=300.0, config_file="Primary Model\mle_lambda_params.json"):
        self.lot_size = lot_size
        self.Q_bar = Q_bar
        self.T = T_horizon
        self.N_t = 1000
        
        self.alpha_max = 5.0
        self.N_alpha = 21
        self.lam_min = 1.0
        self.lam_max = 20.0
        self.N_lam = 10
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                cfg = json.load(f)
            self.kappa = cfg.get('kappa', 1.0)
            self.eta = cfg.get('eta', 0.05)
            self.gamma = cfg.get('gamma', 0.5)
            self.lam_arrival = cfg.get('lam_arrival', 1.0)
            self.lambda_bar = cfg.get('lambda_bar', 10.0)
        else:
            self.kappa = 1.0
            self.eta = 0.05
            self.gamma = 0.5
            self.lam_arrival = 1.0
            self.lambda_bar = 10.0
            
        print(f"Params -> K: {self.kappa:.2f} | Eta: {self.eta:.4f} | Gamma: {self.gamma:.2f} | Lam_Arrival: {self.lam_arrival:.2f}")
        
        print(f"Solving HJB Offline (Horizon={self.T}s)...")
        self.opt_z_ask, self.opt_z_bid, self.opt_mo = solve_hjb_4d(
            float(self.kappa), float(self.eta), float(self.gamma), 
            float(self.lam_arrival), float(self.lambda_bar),
            float(self.lot_size), self.Q_bar, self.T, self.N_t
        )
        
        self.current_alpha = 0.0
        self.last_update_time = 0.0

    def update_alpha(self, time_sec, trade_side, trade_size, current_liquidity_lots):
        dt = max(0, time_sec - self.last_update_time)
        self.current_alpha *= np.exp(-self.kappa * dt)
        self.last_update_time = time_sec
        
        vol_lots = trade_size / self.lot_size
        liq = max(1.0, current_liquidity_lots)
        
        impact = self.eta * (vol_lots / (liq ** self.gamma))
        if trade_side == 'BUY': self.current_alpha += impact  
        elif trade_side == 'SELL': self.current_alpha -= impact

    def get_action(self, time_sec, inventory, time_to_close_sec, current_bid_size, current_ask_size):
        dt = max(0, time_sec - self.last_update_time)
        self.current_alpha *= np.exp(-self.kappa * dt)
        self.last_update_time = time_sec

        if time_to_close_sec > self.T: t_idx = 0 
        else:
            t_idx = int(((self.T - time_to_close_sec) / self.T) * self.N_t)
            t_idx = min(self.N_t - 1, max(0, t_idx))

        # Alpha Index
        d_alpha = (2 * self.alpha_max) / (self.N_alpha - 1)
        a_idx = int(round((self.current_alpha + self.alpha_max) / d_alpha))
        a_idx = min(self.N_alpha - 1, max(0, a_idx))
        
        # Liquidity Index
        current_lam_lots = ((current_bid_size + current_ask_size) / 2.0) / self.lot_size
        d_lam = (self.lam_max - self.lam_min) / (self.N_lam - 1)
        l_idx = int(round((current_lam_lots - self.lam_min) / d_lam))
        l_idx = min(self.N_lam - 1, max(0, l_idx))
        
        # Inventory Index
        inv_lots = int(round(inventory / self.lot_size))
        q_idx = min(2 * self.Q_bar, max(0, inv_lots + self.Q_bar))

        # Look up 4D Policies
        mo = self.opt_mo[t_idx, a_idx, l_idx, q_idx]
        z_ask = self.opt_z_ask[t_idx, a_idx, l_idx, q_idx]
        z_bid = self.opt_z_bid[t_idx, a_idx, l_idx, q_idx]

        return mo, z_ask, z_bid