import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
from scipy.optimize import minimize

@njit
def interp_2d(V_array, a_val, l_val, q_idx, a_max, d_a, N_a, l_min, d_l, N_l):
    f_a = max(0.0, min(N_a - 1.0, (a_val + a_max) / d_a))
    ia0 = int(f_a)
    ia1 = min(ia0 + 1, N_a - 1)
    wa1 = f_a - ia0
    wa0 = 1.0 - wa1
    
    f_l = max(0.0, min(N_l - 1.0, (l_val - l_min) / d_l))
    il0 = int(f_l)
    il1 = min(il0 + 1, N_l - 1)
    wl1 = f_l - il0
    wl0 = 1.0 - wl1
    
    v00 = V_array[ia0, il0, q_idx]
    v01 = V_array[ia0, il1, q_idx]
    v10 = V_array[ia1, il0, q_idx]
    v11 = V_array[ia1, il1, q_idx]
    
    return wa0 * (wl0 * v00 + wl1 * v01) + wa1 * (wl0 * v10 + wl1 * v11)

@njit
def solve_hjb_4d(kappa, eta, gamma, lam_arrival, lambda_bar, lot_size=100.0, Q_bar=50, T=300.0, N_t=3000):
    dt = T / N_t
    N_q = 2 * Q_bar + 1   
    
    alpha_max, N_alpha = 5.0, 15
    d_alpha = (2 * alpha_max) / (N_alpha - 1)
    
    lam_min, lam_max, N_lam = 1.0, 20.0, 5
    d_lam = (lam_max - lam_min) / (N_lam - 1)
    
    phi = 2.5e-6
    psi = 0.001
    half_spread = 0.005              
    upsilon_mo = 0.007
    alpha_scale = 0.001
    
    MAX_CLIP_LOTS = 5 
    V_bins = np.array([1.0, 3.0, 10.0])
    V_probs = np.array([0.7, 0.2, 0.1])
    
    q_grid = np.arange(-Q_bar, Q_bar + 1.0)
    alpha_grid = np.linspace(-alpha_max, alpha_max, N_alpha)
    lam_grid = np.linspace(lam_min, lam_max, N_lam)
    
    V_next = np.zeros((N_alpha, N_lam, N_q))
    V_curr = np.zeros((N_alpha, N_lam, N_q))
    
    N_t_store = N_t // 10
    opt_z_ask = np.zeros((N_t_store, N_alpha, N_lam, N_q), dtype=np.int32)
    opt_z_bid = np.zeros((N_t_store, N_alpha, N_lam, N_q), dtype=np.int32)
    opt_mo_size = np.zeros((N_t_store, N_alpha, N_lam, N_q), dtype=np.int32)
    
    # Terminal Condition
    for i_a in range(N_alpha):
        for i_l in range(N_lam):
            for i_q in range(N_q):
                q = q_grid[i_q]
                V_next[i_a, i_l, i_q] = -np.abs(q) * lot_size * upsilon_mo - psi * (q**2)
            
    # Backward Time Stepping
    for t_idx in range(N_t - 1, -1, -1):
        s_idx = t_idx // 10
        do_store = (t_idx % 10 == 0)
        
        for i_a in range(N_alpha):
            for i_l in range(N_lam):
                for i_q in range(N_q):
                    alpha = alpha_grid[i_a]
                    lam = lam_grid[i_l]
                    q = q_grid[i_q]
                    
                    drift_alpha = 0.0
                    if 0 < i_a < N_alpha - 1:
                        drift_coeff = -kappa * alpha
                        if drift_coeff > 0: drift_alpha = drift_coeff * (V_next[i_a+1, i_l, i_q] - V_next[i_a, i_l, i_q]) / d_alpha
                        else: drift_alpha = drift_coeff * (V_next[i_a, i_l, i_q] - V_next[i_a-1, i_l, i_q]) / d_alpha
                            
                    drift_lam = 0.0
                    if 0 < i_l < N_lam - 1:
                        lam_coeff = 0.5 * (lambda_bar - lam)
                        if lam_coeff > 0: drift_lam = lam_coeff * (V_next[i_a, i_l+1, i_q] - V_next[i_a, i_l, i_q]) / d_lam
                        else: drift_lam = lam_coeff * (V_next[i_a, i_l, i_q] - V_next[i_a, i_l-1, i_q]) / d_lam
                    
                    continuous_pde = drift_alpha + drift_lam - phi * (q**2) + (q * lot_size * alpha * alpha_scale)
                    
                    # Ask Side
                    best_H_ask, best_z_ask = -1e9, 0
                    max_sell_lots = min(MAX_CLIP_LOTS, int(Q_bar + q))
                    for z in range(0, max_sell_lots + 1):
                        expected_fill_val = 0.0
                        for k in range(len(V_bins)):
                            V_mo, prob = V_bins[k], V_probs[k]
                            Y = float(z) * min(1.0, V_mo / lam)
                            
                            alpha_jump = alpha + eta * (V_mo / (lam ** gamma))
                            lam_jump = max(lam_min, lam - V_mo)
                            
                            q_exact = i_q - Y
                            q_f, q_c = int(np.floor(q_exact)), int(np.floor(q_exact)) + 1
                            w_c = q_exact - q_f
                            
                            V_jumped_f = interp_2d(V_next, alpha_jump, lam_jump, min(N_q-1, max(0, q_f)), alpha_max, d_alpha, N_alpha, lam_min, d_lam, N_lam)
                            V_jumped_c = interp_2d(V_next, alpha_jump, lam_jump, min(N_q-1, max(0, q_c)), alpha_max, d_alpha, N_alpha, lam_min, d_lam, N_lam)
                            
                            val_jump = ((1.0 - w_c) * V_jumped_f + w_c * V_jumped_c) - V_next[i_a, i_l, i_q] + (Y * lot_size * half_spread)
                            expected_fill_val += prob * val_jump
                            
                        H_ask = (lam_arrival / 2.0) * expected_fill_val
                        if H_ask > best_H_ask: best_H_ask, best_z_ask = H_ask, z
                                
                    # Bid Side
                    best_H_bid, best_z_bid = -1e9, 0
                    max_buy_lots = min(MAX_CLIP_LOTS, int(Q_bar - q))
                    for z in range(0, max_buy_lots + 1):
                        expected_fill_val = 0.0
                        for k in range(len(V_bins)):
                            V_mo, prob = V_bins[k], V_probs[k]
                            Y = float(z) * min(1.0, V_mo / lam)
                            
                            alpha_jump = alpha - eta * (V_mo / (lam ** gamma))
                            lam_jump = max(lam_min, lam - V_mo)
                            
                            q_exact = i_q + Y
                            q_f, q_c = int(np.floor(q_exact)), int(np.floor(q_exact)) + 1
                            w_c = q_exact - q_f
                            
                            V_jumped_f = interp_2d(V_next, alpha_jump, lam_jump, min(N_q-1, max(0, q_f)), alpha_max, d_alpha, N_alpha, lam_min, d_lam, N_lam)
                            V_jumped_c = interp_2d(V_next, alpha_jump, lam_jump, min(N_q-1, max(0, q_c)), alpha_max, d_alpha, N_alpha, lam_min, d_lam, N_lam)
                            
                            val_jump = ((1.0 - w_c) * V_jumped_f + w_c * V_jumped_c) - V_next[i_a, i_l, i_q] + (Y * lot_size * half_spread)
                            expected_fill_val += prob * val_jump
                            
                        H_bid = (lam_arrival / 2.0) * expected_fill_val
                        if H_bid > best_H_bid: best_H_bid, best_z_bid = H_bid, z
                    
                    continuous_pde += best_H_ask + best_H_bid
                    
                    # Impulse Control
                    M_val, best_mo_size = -1e9, 0
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
                        if do_store:
                            opt_mo_size[s_idx, i_a, i_l, i_q], opt_z_ask[s_idx, i_a, i_l, i_q], opt_z_bid[s_idx, i_a, i_l, i_q] = best_mo_size, 0, 0
                    else:
                        V_curr[i_a, i_l, i_q] = V_unconstrained
                        if do_store:
                            opt_mo_size[s_idx, i_a, i_l, i_q], opt_z_ask[s_idx, i_a, i_l, i_q], opt_z_bid[s_idx, i_a, i_l, i_q] = 0, best_z_ask, best_z_bid
                        
        V_next[:] = V_curr[:]
        
    return opt_z_ask, opt_z_bid, opt_mo_size

@njit
def fast_backtest_engine(
    times, event_types, bids, asks, bid_sizes, ask_sizes, trade_prices, trade_sizes, trade_sides,
    opt_z_ask, opt_z_bid, opt_mo,
    kappa, eta, gamma, Q_bar, lot_size, T_horizon, N_t_store,
    alpha_max, N_alpha, lam_min, lam_max, N_lam
):
    N_events = len(times)
    
    max_snaps = 1000 
    pnl_hist = np.zeros(max_snaps)
    time_hist = np.zeros(max_snaps)
    inv_hist = np.zeros(max_snaps)
    snap_idx = 0
    
    inventory = 0
    cash = 0.0
    current_alpha = 0.0
    last_update_time = times[0]
    
    current_bid = 0.0
    current_ask = 0.0
    current_bidsiz_shares = 100.0
    current_asksiz_shares = 100.0
    
    active_z_bid = 0
    active_z_ask = 0
    
    MARKET_OPEN_SEC = 9.5 * 3600
    WARMUP_SEC = 300.0
    MARKET_CLOSE_SEC = 16.0 * 3600
    MAX_SHARES = Q_bar * lot_size
    
    total_volume_traded = 0
    
    d_alpha = (2 * alpha_max) / (N_alpha - 1)
    d_lam = (lam_max - lam_min) / (N_lam - 1)
    
    last_save_time = -1.0
    
    for i in range(N_events):
        t_sec = times[i]
        ev_type = event_types[i]
        time_to_close = MARKET_CLOSE_SEC - t_sec
        
        # End of day liquidation
        if time_to_close < 10.0:
            if inventory > 0: cash += inventory * current_bid
            elif inventory < 0: cash -= abs(inventory) * current_ask
            inventory = 0
            
            pnl_hist[snap_idx] = cash
            time_hist[snap_idx] = t_sec
            inv_hist[snap_idx] = inventory
            snap_idx += 1
            break

        # Alpha decay update
        dt = max(0.0, t_sec - last_update_time)
        current_alpha *= np.exp(-kappa * dt)
        last_update_time = t_sec

        if ev_type == 0:
            if bids[i] != current_bid or asks[i] != current_ask:
                active_z_bid, active_z_ask = 0, 0
                
            current_bid, current_ask = bids[i], asks[i]
            current_bidsiz_shares = max(100.0, bid_sizes[i] * 100.0)
            current_asksiz_shares = max(100.0, ask_sizes[i] * 100.0)
            
            if t_sec >= MARKET_OPEN_SEC + WARMUP_SEC:
                t_idx = 0 if time_to_close > T_horizon else int(((T_horizon - time_to_close) / T_horizon) * N_t_store)
                t_idx = min(N_t_store - 1, max(0, t_idx))
                
                a_idx = int(round((current_alpha + alpha_max) / d_alpha))
                a_idx = min(N_alpha - 1, max(0, a_idx))
                
                current_lam_lots = ((current_bidsiz_shares + current_asksiz_shares) / 2.0) / lot_size
                l_idx = int(round((current_lam_lots - lam_min) / d_lam))
                l_idx = min(N_lam - 1, max(0, l_idx))
                
                inv_lots = int(round(inventory / lot_size))
                q_idx = min(2 * Q_bar, max(0, inv_lots + Q_bar))

                mo = opt_mo[t_idx, a_idx, l_idx, q_idx]
                z_ask = opt_z_ask[t_idx, a_idx, l_idx, q_idx]
                z_bid = opt_z_bid[t_idx, a_idx, l_idx, q_idx]
                
                if mo != 0:
                    side_is_buy = (mo > 0)
                    size = abs(mo) * lot_size
                    if side_is_buy and inventory < 0:
                        size = min(size, abs(inventory))
                        cash -= size * (current_ask + 0.002) 
                        inventory += size
                        total_volume_traded += size
                    elif not side_is_buy and inventory > 0:
                        size = min(size, inventory)
                        cash += size * (current_bid - 0.002) 
                        inventory -= size
                        total_volume_traded += size
                    active_z_bid, active_z_ask = 0, 0
                else:
                    active_z_ask = z_ask * lot_size
                    active_z_bid = z_bid * lot_size

        else:
            trade_price, trade_size, trade_side = trade_prices[i], trade_sizes[i], trade_sides[i]
            if trade_side != 0:
                prevailing_liquidity = current_asksiz_shares if trade_side == 1 else current_bidsiz_shares
                vol_lots = trade_size / lot_size
                liq_lots = max(1.0, prevailing_liquidity / lot_size)
                impact = eta * (vol_lots / (liq_lots ** gamma))
                if trade_side == 1: current_alpha += impact
                else: current_alpha -= impact
                
                if t_sec >= MARKET_OPEN_SEC + WARMUP_SEC:
                    max_can_buy = MAX_SHARES - inventory
                    max_can_sell = MAX_SHARES + inventory
                    
                    if trade_side == 1 and trade_price >= current_ask and active_z_ask > 0:
                        consumption_ratio = min(1.0, trade_size / current_asksiz_shares)
                        filled_qty = active_z_ask * consumption_ratio 
                        filled_qty = min(filled_qty, trade_size) 
                        if filled_qty >= 1.0:
                            filled_qty = min(int(filled_qty), max_can_sell)
                            inventory -= filled_qty
                            cash += filled_qty * current_ask
                            active_z_ask -= filled_qty
                            total_volume_traded += filled_qty
                            
                    elif trade_side == -1 and trade_price <= current_bid and active_z_bid > 0:
                        consumption_ratio = min(1.0, trade_size / current_bidsiz_shares)
                        filled_qty = active_z_bid * consumption_ratio
                        filled_qty = min(filled_qty, trade_size) 
                        if filled_qty >= 1.0:
                            filled_qty = min(int(filled_qty), max_can_buy)
                            inventory += filled_qty
                            cash -= filled_qty * current_bid
                            active_z_bid -= filled_qty
                            total_volume_traded += filled_qty

        if last_save_time == -1.0 or (t_sec - last_save_time) >= 60.0:
            mid = (current_bid + current_ask) / 2.0
            unrealized = inventory * mid if mid > 0 else 0
            pnl_hist[snap_idx] = cash + unrealized
            time_hist[snap_idx] = t_sec
            inv_hist[snap_idx] = inventory
            snap_idx += 1
            last_save_time = t_sec

    return pnl_hist[:snap_idx], time_hist[:snap_idx], inv_hist[:snap_idx], total_volume_traded

def parse_time(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def load_and_prep_taq_fast(quotes_file, trades_file):
    quotes = pd.read_csv(quotes_file)
    trades = pd.read_csv(trades_file)
    
    quotes['TIME_SEC'] = quotes['TIME_M'].apply(parse_time)
    trades['TIME_SEC'] = trades['TIME_M'].apply(parse_time)
    quotes = quotes[quotes['ASK'] > quotes['BID']].copy()
    
    quotes['EVENT_TYPE'] = 0
    trades['EVENT_TYPE'] = 1
    
    for col in ['PRICE', 'SIZE']: quotes[col] = 0.0
    for col in ['BID', 'ASK', 'BIDSIZ', 'ASKSIZ']: trades[col] = 0.0
    
    events = pd.concat([quotes, trades]).sort_values(by=['TIME_SEC', 'EVENT_TYPE']).reset_index(drop=True)
    events['BID'] = events['BID'].replace(0.0, np.nan).ffill().fillna(0)
    events['ASK'] = events['ASK'].replace(0.0, np.nan).ffill().fillna(0)
    events['BIDSIZ'] = events['BIDSIZ'].replace(0.0, np.nan).ffill().fillna(0)
    events['ASKSIZ'] = events['ASKSIZ'].replace(0.0, np.nan).ffill().fillna(0)
    
    events['MID'] = (events['BID'] + events['ASK']) / 2.0
    
    conditions = [events['PRICE'] > events['MID'], events['PRICE'] < events['MID']]
    choices = [1, -1]
    events['SIDE'] = np.select(conditions, choices, default=0)
    
    return (
        events['TIME_SEC'].values.astype(np.float64),
        events['EVENT_TYPE'].values.astype(np.int8),
        events['BID'].values.astype(np.float64),
        events['ASK'].values.astype(np.float64),
        events['BIDSIZ'].values.astype(np.float64),
        events['ASKSIZ'].values.astype(np.float64),
        events['PRICE'].values.astype(np.float64),
        events['SIZE'].values.astype(np.float64),
        events['SIDE'].values.astype(np.int8)
    )

def calculate_metrics(pnl_series, total_volume, days_count):
    returns = pnl_series.diff().dropna()
    total_pnl = pnl_series.iloc[-1] if not pnl_series.empty else 0
    
    # Sharpe Ratio
    sharpe = np.sqrt(390 * 252) * (returns.mean() / returns.std()) if not returns.empty and returns.std() > 0 else 0.0
    
    # Sortino Ratio
    downside = returns[returns < 0]
    sortino = np.sqrt(390 * 252) * (returns.mean() / downside.std()) if not downside.empty and downside.std() > 0 else 0.0
    
    # Max Drawdown
    cum_max = pnl_series.cummax()
    drawdown = cum_max - pnl_series
    max_drawdown = drawdown.max()
    
    # Win Rate
    win_rate = (returns > 0).mean() * 100 if not returns.empty else 0.0
    
    # Calmar Ratio
    annualized_return = (total_pnl / days_count) * 252
    calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
    
    print("\n" + "="*40)
    print("      ADVANCED BACKTEST METRICS      ")
    print("="*40)
    print(f"Total Trading Days : {days_count}")
    print(f"Total Net PnL      : ${total_pnl:.2f}")
    print(f"Avg Daily PnL      : ${(total_pnl/days_count):.2f}")
    print(f"Total Vol Traded   : {total_volume:,} Shares")
    print(f"PnL Per Share      : ${(total_pnl/max(1, total_volume)):.4f}")
    print("-" * 40)
    print(f"Ann. Sharpe Ratio  : {sharpe:.2f}")
    print(f"Ann. Sortino Ratio : {sortino:.2f}")
    print(f"Calmar Ratio       : {calmar:.2f}")
    print(f"Max Drawdown       : ${max_drawdown:.2f}")
    print(f"Minute Win Rate    : {win_rate:.1f}%")
    print("="*40)

@njit
def calculate_lambda_log_likelihood(kappa, eta, gamma, theta, 
                                    mo_buy_times, mo_buy_vols, mo_buy_liqs,
                                    mo_sell_times, mo_sell_vols, mo_sell_liqs,
                                    jump_up_times, jump_down_times, 
                                    T_max, dt=0.1):
    N_steps = int(T_max / dt)
    alpha = np.zeros(N_steps)
    
    mo_buy_idx = 0
    mo_sell_idx = 0
    current_alpha = 0.0
    
    for i in range(1, N_steps):
        t_start = (i - 1) * dt
        t_end = i * dt
        current_alpha *= np.exp(-kappa * dt)
        
        while mo_buy_idx < len(mo_buy_times) and mo_buy_times[mo_buy_idx] < t_end:
            v = mo_buy_vols[mo_buy_idx]
            l = max(1.0, mo_buy_liqs[mo_buy_idx])
            impact = eta * (v / (l ** gamma))
            current_alpha += impact
            mo_buy_idx += 1
            
        while mo_sell_idx < len(mo_sell_times) and mo_sell_times[mo_sell_idx] < t_end:
            v = mo_sell_vols[mo_sell_idx]
            l = max(1.0, mo_sell_liqs[mo_sell_idx])
            impact = eta * (v / (l ** gamma))
            current_alpha -= impact
            mo_sell_idx += 1
            
        alpha[i] = current_alpha

    mu_plus = np.maximum(alpha, 0.0) + theta
    mu_minus = np.maximum(-alpha, 0.0) + theta
    integral_penalty = np.sum(mu_plus) * dt + np.sum(mu_minus) * dt
    
    log_rewards = 0.0
    for t_up in jump_up_times:
        idx = int(t_up / dt)
        if idx < N_steps:
            intensity = max(alpha[idx], 0.0) + theta
            log_rewards += np.log(intensity)
            
    for t_down in jump_down_times:
        idx = int(t_down / dt)
        if idx < N_steps:
            intensity = max(-alpha[idx], 0.0) + theta
            log_rewards += np.log(intensity)
            
    return -(log_rewards - integral_penalty)
    
def recalibrate_daily_model(times, event_types, bids, asks, bid_sizes, ask_sizes, trade_sides, trade_sizes, current_params, lot_size=100.0):
    # Extract Trade Data
    trades_mask = (event_types == 1)
    buy_mask = trades_mask & (trade_sides == 1)
    sell_mask = trades_mask & (trade_sides == -1)
    
    mo_buy_times = times[buy_mask]
    mo_buy_vols = trade_sizes[buy_mask] / lot_size
    mo_buy_liqs = ask_sizes[buy_mask] / lot_size
    
    mo_sell_times = times[sell_mask]
    mo_sell_vols = trade_sizes[sell_mask] / lot_size
    mo_sell_liqs = bid_sizes[sell_mask] / lot_size
    
    # Extract Quote Jumps
    quotes_mask = (event_types == 0)
    q_times = times[quotes_mask]
    q_mids = (bids[quotes_mask] + asks[quotes_mask]) / 2.0
    
    mids_diff = np.diff(q_mids)
    jump_up_times = q_times[1:][mids_diff > 0]
    jump_down_times = q_times[1:][mids_diff < 0]
    
    T_max = times[-1] + 1.0
    total_time = T_max - times[0]
    
    # Fast Empirical Estimates
    lam_arrival = np.sum(trades_mask) / total_time
    lambda_bar = np.mean((bid_sizes[quotes_mask] + ask_sizes[quotes_mask]) / (2.0 * lot_size))
    
    # MLE Optimizer
    def objective(params):
        kappa, eta, gamma, theta = params
        if kappa <= 0 or eta <= 0 or gamma < 0 or gamma > 1 or theta <= 1e-6: return 1e9
        return calculate_lambda_log_likelihood(
            kappa, eta, gamma, theta, 
            mo_buy_times, mo_buy_vols, mo_buy_liqs, 
            mo_sell_times, mo_sell_vols, mo_sell_liqs, 
            jump_up_times, jump_down_times, T_max
        )
        
    initial_guess = [current_params['kappa'], current_params['eta'], current_params['gamma'], 0.1]
    bounds =[(0.01, 100.0), (0.001, 5.0), (0.0, 1.0), (0.001, 5.0)]
    res = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
    
    if res.success:
        return {
            'kappa': res.x[0], 'eta': res.x[1], 'gamma': res.x[2],
            'lam_arrival': lam_arrival, 'lambda_bar': lambda_bar, 'success': True
        }
    return {'success': False}

def run_multi_day_backtest(dates_list, quotes_dir, trades_dir, ticker="KO"):
    params = {'kappa': 1.0, 'eta': 0.05, 'gamma': 0.5, 'lam_arrival': 1.0, 'lambda_bar': 10.0}
    Q_bar, lot_size, T_horizon = 50, 100, 300.0
    N_t, N_alpha, N_lam = 3000, 15, 5
    alpha_max, lam_min, lam_max = 5.0, 1.0, 20.0
    
    print("Solving Initial PDE for Day 1...")
    opt_z_ask, opt_z_bid, opt_mo = solve_hjb_4d(
        params['kappa'], params['eta'], params['gamma'], params['lam_arrival'], params['lambda_bar'],
        float(lot_size), Q_bar, T_horizon, N_t
    )
    N_t_store = N_t // 10
    
    agg_pnl, agg_time, agg_inv =[], [],[]
    total_volume, cumulative_pnl_offset, time_offset, valid_days = 0, 0.0, 0.0, 0
    
    for date_str in dates_list:
        q_file = os.path.join(quotes_dir, f"{ticker}_{date_str}_quotes.csv")
        t_file = os.path.join(trades_dir, f"{ticker}_{date_str}_trades.csv")
        
        if not os.path.exists(q_file) or not os.path.exists(t_file):
            continue
            
        print(f"\n--- Trading {date_str} ---")
        print(f"Current Params -> K: {params['kappa']:.3f} | Eta: {params['eta']:.4f} | Gamma: {params['gamma']:.2f}")
        arrays = load_and_prep_taq_fast(q_file, t_file)
        
        p_hist, t_hist, i_hist, d_vol = fast_backtest_engine(
            *arrays, opt_z_ask, opt_z_bid, opt_mo,
            params['kappa'], params['eta'], params['gamma'], 
            Q_bar, lot_size, T_horizon, N_t_store,
            alpha_max, N_alpha, lam_min, lam_max, N_lam
        )
        
        agg_pnl.extend((p_hist + cumulative_pnl_offset).tolist())
        agg_time.extend((t_hist + time_offset).tolist())
        agg_inv.extend(i_hist.tolist())
        total_volume += d_vol
        cumulative_pnl_offset += p_hist[-1]
        time_offset += 24 * 3600
        valid_days += 1
        
        new_est = recalibrate_daily_model(arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5], arrays[8], arrays[7], params, lot_size)
        
        if new_est['success']:
            alpha_struct = 0.02  
            params['kappa'] = (1 - alpha_struct)*params['kappa'] + alpha_struct*new_est['kappa']
            params['eta'] = (1 - alpha_struct)*params['eta'] + alpha_struct*new_est['eta']
            params['gamma'] = (1 - alpha_struct)*params['gamma'] + alpha_struct*new_est['gamma']
            
            alpha_obs = 0.10 
            safe_lam = min(2.5, new_est['lam_arrival']) 
            
            params['lam_arrival'] = (1 - alpha_obs)*params['lam_arrival'] + alpha_obs*safe_lam
            params['lambda_bar'] = (1 - alpha_obs)*params['lambda_bar'] + alpha_obs*new_est['lambda_bar']
            
            print("Re-solving HJB PDE with new parameters...")
            opt_z_ask, opt_z_bid, opt_mo = solve_hjb_4d(
                params['kappa'], params['eta'], params['gamma'], params['lam_arrival'], params['lambda_bar'],
                float(lot_size), Q_bar, T_horizon, N_t
            )

    if valid_days > 0:
        calculate_metrics(pd.Series(agg_pnl), total_volume, valid_days)
        
        plt.figure(figsize=(14, 7))
        plt.subplot(2, 1, 1)
        plt.plot(agg_time, agg_pnl, color='green', linewidth=1.5)
        plt.title('Multi-Day Walk-Forward Adaptive Market Maker PnL')
        plt.ylabel('Cumulative PnL ($)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(agg_time, agg_inv, color='orange', linewidth=1)
        plt.ylabel('Inventory (Shares)')
        plt.xlabel('Continuous Time (Seconds across days)')
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    else:
        print("No valid days were processed.")

if __name__ == "__main__":
    quotes_directory = r"Extracted_Market_Data\\KO\\KO_Quotes"
    trades_directory = r"Extracted_Market_Data\\KO\\KO_Trades"
    
    dates_to_run =["20251001", "20251002", "20251003"] # Add more data as needed, but computation time scales proportionally with the number of days
    
    run_multi_day_backtest(dates_to_run, quotes_directory, trades_directory, ticker="KO")