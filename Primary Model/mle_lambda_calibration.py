import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numba import njit
import json
import os

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
    
    # Reconstruct Alpha Signal
    current_alpha = 0.0
    for i in range(1, N_steps):
        t_start = (i - 1) * dt
        t_end = i * dt
        
        # Exponential Decay
        current_alpha *= np.exp(-kappa * dt)
        
        # Add impacts from Buy Market Orders
        while mo_buy_idx < len(mo_buy_times) and mo_buy_times[mo_buy_idx] < t_end:
            v = mo_buy_vols[mo_buy_idx]
            l = max(1.0, mo_buy_liqs[mo_buy_idx])
            
            impact = eta * (v / (l ** gamma))
            current_alpha += impact
            mo_buy_idx += 1
            
        # Add impacts from Sell Market Orders
        while mo_sell_idx < len(mo_sell_times) and mo_sell_times[mo_sell_idx] < t_end:
            v = mo_sell_vols[mo_sell_idx]
            l = max(1.0, mo_sell_liqs[mo_sell_idx])
            
            impact = eta * (v / (l ** gamma))
            current_alpha -= impact
            mo_sell_idx += 1
            
        alpha[i] = current_alpha

    # Integral Penalty 
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


def calibrate_lambda_model(events_df, output_file="mle_lambda_params.json"):
    print("Extracting event timestamps, volumes, and prevailing liquidity...")
    
    trades = events_df[events_df['EVENT_TYPE'] == 'TRADE'].copy()
    
    buy_trades = trades[trades['SIDE'] == 'BUY']
    sell_trades = trades[trades['SIDE'] == 'SELL']
    
    mo_buy_times = buy_trades['TIME_SEC'].values
    mo_buy_vols = (buy_trades['SIZE'] / 100.0).values
    mo_buy_liqs = (buy_trades['ASKSIZ'] / 100.0).values # Buys eat Ask liquidity
    
    mo_sell_times = sell_trades['TIME_SEC'].values
    mo_sell_vols = (sell_trades['SIZE'] / 100.0).values
    mo_sell_liqs = (sell_trades['BIDSIZ'] / 100.0).values # Sells eat Bid liquidity
    
    # Extract Mid-Price Jumps
    quotes = events_df[events_df['EVENT_TYPE'] == 'QUOTE'].copy()
    quotes['PREV_MID'] = quotes['MID'].shift(1)
    
    jumps_up = quotes[quotes['MID'] > quotes['PREV_MID']]['TIME_SEC'].values
    jumps_down = quotes[quotes['MID'] < quotes['PREV_MID']]['TIME_SEC'].values
    
    T_max = events_df['TIME_SEC'].max() + 1.0
    
    def objective(params):
        kappa, eta, gamma, theta = params
        if kappa <= 0 or eta <= 0 or gamma < 0 or gamma > 1 or theta <= 1e-6:
            return 1e9
            
        return calculate_lambda_log_likelihood(
            kappa, eta, gamma, theta,
            mo_buy_times, mo_buy_vols, mo_buy_liqs,
            mo_sell_times, mo_sell_vols, mo_sell_liqs,
            jumps_up, jumps_down, T_max
        )

    initial_guess =[1.0, 0.05, 0.5, 0.1]
    bounds =[(0.01, 100.0), (0.001, 5.0), (0.0, 1.0), (0.001, 5.0)]
    
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
    
    if result.success:
        total_time = T_max - events_df['TIME_SEC'].min()
        lam_arrival = len(trades) / total_time
        
        # Calculate Average Liquidity (lambda_bar) for the PDE drift
        lambda_bar = (quotes['BIDSIZ'].mean() + quotes['ASKSIZ'].mean()) / 200.0
        
        params_dict = {
            "kappa": float(result.x[0]),
            "eta": float(result.x[1]),
            "gamma": float(result.x[2]),
            "theta": float(result.x[3]),
            "lam_arrival": float(lam_arrival),
            "lambda_bar": float(lambda_bar)
        }
        
        with open(output_file, "w") as f:
            json.dump(params_dict, f, indent=4)
            
        print("\n=== MLE CALIBRATION SUCCESS ===")
        print(f"Optimal Kappa (Decay)          : {params_dict['kappa']:.4f}")
        print(f"Optimal Eta (Base Impact)      : {params_dict['eta']:.4f}")
        print(f"Optimal Gamma (Liquidity Damp) : {params_dict['gamma']:.4f}")
        print(f"Optimal Theta (Baseline Noise) : {params_dict['theta']:.4f}")
        print(f"Empirical MO Arrival Rate      : {params_dict['lam_arrival']:.4f} orders/sec")
        print(f"Empirical Average L1 Liquidity : {params_dict['lambda_bar']:.2f} lots")
        print(f"Saved to {output_file}")
    else:
        print("MLE Failed to converge.")

if __name__ == "__main__":
    from Primary_backtester import load_and_prep_taq
    
    # TEST ON DAY 1 DATA
    q_file = r"Extracted_Market_Data\KO\KO_Quotes\KO_20251001_quotes.csv"
    t_file = r"Extracted_Market_Data\KO\KO_Trades\KO_20251001_trades.csv"
    
    events_df = load_and_prep_taq(q_file, t_file)
    calibrate_lambda_model(events_df)