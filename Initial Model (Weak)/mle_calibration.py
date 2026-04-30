import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numba import njit
import json
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON_PATH = os.path.join(CURRENT_DIR, "mle_params.json")

@njit
def exact_log_likelihood(kappa, eta_plus, eta_minus, theta, 
                         event_times, event_types, T_max):
    current_alpha = 0.0
    last_time = 0.0
    
    log_rewards = 0.0
    integral_penalty = 0.0
    
    for i in range(len(event_times)):
        t = event_times[i]
        e_type = event_types[i]
        
        dt = t - last_time

        alpha_integral = abs(current_alpha) * (1.0 - np.exp(-kappa * dt)) / kappa
        integral_penalty += (2.0 * theta * dt) + alpha_integral
        
        current_alpha *= np.exp(-kappa * dt)
        
        # 3. Process the Event
        if e_type == 3: # Mid-price Jump UP
            intensity = max(current_alpha, 0.0) + theta
            log_rewards += np.log(intensity)
            
        elif e_type == 4: # Mid-price Jump DOWN
            intensity = max(-current_alpha, 0.0) + theta
            log_rewards += np.log(intensity)
            
        elif e_type == 1:
            current_alpha += eta_plus
            
        elif e_type == 2:
            current_alpha -= eta_minus
            
        last_time = t

    dt_final = T_max - last_time
    if dt_final > 0:
        alpha_integral = abs(current_alpha) * (1.0 - np.exp(-kappa * dt_final)) / kappa
        integral_penalty += (2.0 * theta * dt_final) + alpha_integral
        
    return -(log_rewards - integral_penalty)

def calibrate_and_save(events_df, output_file=DEFAULT_JSON_PATH):
    
    trades = events_df[events_df['EVENT_TYPE'] == 'TRADE'].copy()
    grouped_trades = trades.groupby(['TIME_SEC', 'SIDE']).size().reset_index()
    
    mo_buy_times = grouped_trades[grouped_trades['SIDE'] == 'BUY']['TIME_SEC'].values
    mo_sell_times = grouped_trades[grouped_trades['SIDE'] == 'SELL']['TIME_SEC'].values
    
    quotes = events_df[events_df['EVENT_TYPE'] == 'QUOTE'].copy()
    quotes = quotes.drop_duplicates(subset=['TIME_SEC'], keep='last').copy()
    quotes['PREV_MID'] = quotes['MID'].shift(1)
    
    jumps_up = quotes[quotes['MID'] > quotes['PREV_MID']]['TIME_SEC'].values
    jumps_down = quotes[quotes['MID'] < quotes['PREV_MID']]['TIME_SEC'].values
    
    T_max = events_df['TIME_SEC'].max() + 1.0

    all_events =[]
    for t in mo_buy_times: all_events.append((t, 1))
    for t in mo_sell_times: all_events.append((t, 2))
    for t in jumps_up: all_events.append((t, 3))
    for t in jumps_down: all_events.append((t, 4))
    
    all_events.sort(key=lambda x: (x[0], x[1]))
    
    event_times = np.array([x[0] for x in all_events])
    event_types = np.array([x[1] for x in all_events], dtype=np.int32)
    
    def objective(params):
        kappa, eta_plus, eta_minus, theta = params
        if kappa <= 1e-6 or eta_plus <= 1e-6 or eta_minus <= 1e-6 or theta <= 1e-6:
            return 1e9 # Enforce strictly positive constraints
            
        return exact_log_likelihood(
            kappa, eta_plus, eta_minus, theta,
            event_times, event_types, T_max
        )

    initial_guess =[5.0, 0.05, 0.05, 0.1]
    bounds =[(0.1, 200.0), (0.001, 10.0), (0.001, 10.0), (0.001, 10.0)]
    
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-9})
    
    if result.success:
        total_time = T_max - events_df['TIME_SEC'].min()
        lam_plus = len(mo_buy_times) / total_time
        lam_minus = len(mo_sell_times) / total_time
        
        avg_eta = (result.x[1] + result.x[2]) / 2.0
        avg_lam = (lam_plus + lam_minus) / 2.0
        
        params_dict = {
            "kappa": float(result.x[0]),
            "eta": float(avg_eta),
            "theta": float(result.x[3]),
            "lam_arrival": float(avg_lam)
        }
        
        with open(output_file, "w") as f:
            json.dump(params_dict, f, indent=4)
        print(json.dumps(params_dict, indent=4))
    else:
        print("MLE Failed to converge.")
        print(result.message)

if __name__ == "__main__":
    from Momentum_Backtester import load_and_prep_taq
    q_file = r"Extracted_Market_Data\\IBM\\IBM_Quotes\\IBM_20251001_quotes.csv"
    t_file = r"Extracted_Market_Data\\IBM\\IBM_Trades\\IBM_20251001_trades.csv"
    events_df = load_and_prep_taq(q_file, t_file)
    calibrate_and_save(events_df)