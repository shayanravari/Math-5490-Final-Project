import multiprocessing as mp
import pandas as pd
import numpy as np
import os
import json
import glob
from scipy.optimize import curve_fit

try:
    import tick.base.base
    tick.base.base.Base.__setattr__ = object.__setattr__
    
    import tick.solver.history.history
    tick.solver.history.history.History.__setattr__ = object.__setattr__
    tick.solver.history.history.History._minimum_col_width = 9
    
    import tick.solver.base.solver
    tick.solver.base.solver.Solver.__setattr__ = object.__setattr__
    
    import tick.hawkes.inference.base.learner_hawkes_noparam
    tick.hawkes.inference.base.learner_hawkes_noparam.LearnerHawkesNoParam.__setattr__ = object.__setattr__
    
    import tick.hawkes.inference.hawkes_em
    tick.hawkes.inference.hawkes_em.HawkesEM.__setattr__ = object.__setattr__
    
    from tick.hawkes import HawkesEM
    TICK_AVAILABLE = True
except ImportError:
    TICK_AVAILABLE = False
except Exception as e:
    TICK_AVAILABLE = False

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(ROOT_DIR, "Extracted_Market_Data")
TICKERS = ["IBM", "KO", "TGT"]

def power_law(t, a, g, b):
    return a * (1 + g * t)**(-b)

def calibrate_ticker(ticker):
    base_dir = os.path.join(OUT_DIR, ticker)
    event_file = os.path.join(base_dir, f"{ticker}_Hawkes_Events.csv")
    config_file = os.path.join(base_dir, f"{ticker}_config.json")
    
    print(f"\n{'='*50}\nCALIBRATING {ticker}\n{'='*50}")
    if not os.path.exists(event_file):
        print(f"Events file not found for {ticker}.")
        return

    df = pd.read_csv(event_file)
    config = {}

    trade_files = sorted(glob.glob(os.path.join(base_dir, f"{ticker}_Trades", "*_trades.csv")))
    if trade_files:
        first_trade = pd.read_csv(trade_files[0], nrows=1)
        initial_price = float(first_trade['PRICE'].iloc[0])
    else:
        initial_price = 100.00
    config['initial_price'] = initial_price

    quote_files = sorted(glob.glob(os.path.join(base_dir, f"{ticker}_Quotes", "*_quotes.csv")))
    if quote_files:
        q_df = pd.read_csv(quote_files[0], usecols=['BID', 'ASK'])
        q_df = q_df[(q_df['BID'] > 0) & (q_df['ASK'] > q_df['BID'])]
        mean_spread_ticks = ((q_df['ASK'] - q_df['BID']) / 0.01).mean()
        
        if mean_spread_ticks > 2.0:
            beta_spread = np.log(2.0) / np.log(mean_spread_ticks - 1.0)
            beta_spread = max(0.1, min(beta_spread, 1.5))
        else:
            beta_spread = 1.5
        empty_prob = max(0.0, 1.0 - (1.0 / mean_spread_ticks)) if mean_spread_ticks >= 1.0 else 0.0
    else:
        beta_spread = 1.0
        mean_spread_ticks = 1.0
        empty_prob = 0.0

    config['beta_spread'] = round(float(beta_spread), 3)
    config['empty_level_prob'] = round(float(empty_prob), 3) # Save it to the JSON
    print(f"[{ticker}] True Avg Spread: {mean_spread_ticks:.2f} ticks | Empty Level Prob: {empty_prob*100:.1f}%")

    def calibrate_size_distribution(sizes_series):
        if sizes_series.empty:
            return {'round_numbers': [100], 'round_probs':[1.0], 'geometric_p': 0.05, 'p_round': 1.0}
            
        sizes = sizes_series.values
        total_orders = len(sizes)
        
        size_counts = sizes_series.value_counts()
        top_sizes = size_counts.head(10).index.astype(int).tolist()
        top_sizes.sort()
        
        round_counts = {rn: int(size_counts.loc[rn]) for rn in top_sizes}
        total_round = sum(round_counts.values())
        
        non_round_df = sizes_series[~sizes_series.isin(top_sizes)]
        clean_non_round_df = non_round_df[non_round_df <= 5000]
        mean_non_round = clean_non_round_df.mean() if not clean_non_round_df.empty else 100.0
        
        return {
            'round_numbers': top_sizes,
            'round_probs':[float(round_counts[rn]/total_round) for rn in top_sizes] if total_round > 0 else[1.0],
            'geometric_p': float(min(1.0, 1.0 / mean_non_round)),
            'p_round': float(total_round / total_orders) if total_orders > 0 else 0.85
        }
    
    lo_events = df[df['event_id'].isin([0, 3, 4, 5])]['size']
    mo_events = df[df['event_id'].isin([2, 7])]['size']
    co_events = df[df['event_id'].isin([1, 6])]['size']
    
    config['lo_size_model'] = calibrate_size_distribution(lo_events)
    config['mo_size_model'] = calibrate_size_distribution(mo_events)
    config['co_size_model'] = calibrate_size_distribution(co_events)
    
    print(f"[{ticker}] Discovered LO Sizes: {config['lo_size_model']['round_numbers']}")
    print(f"[{ticker}] Discovered MO Sizes: {config['mo_size_model']['round_numbers']}")
    print(f"[{ticker}] Discovered CO Sizes: {config['co_size_model']['round_numbers']}")

    num_days = max(1, df['date'].nunique())
    total_sim_seconds = num_days * (6.5 * 3600)
    
    lambda_hat = np.zeros(8)
    event_counts = df['event_id'].value_counts().to_dict()
    for i in range(8):
        lambda_hat[i] = event_counts.get(i, 0) / total_sim_seconds

    df['bin'] = ((df['time'] - 34200) // 1800).astype(int)
    bin_counts = df[df['bin'] <= 12]['bin'].value_counts().sort_index()
    avg_per_bin = bin_counts.mean()
    u_shape = {str(int(b)): round(float(c / avg_per_bin), 2) for b, c in bin_counts.items()}
    config['u_shape_bins'] = u_shape
    max_f_Qt = max(u_shape.values()) if u_shape else 1.0

    alpha = np.zeros((8, 8))
    gamma = np.ones((8, 8)) 
    beta = np.full((8, 8), 1.5) 
    M_hat = np.zeros((8, 8)) 
    
    if TICK_AVAILABLE:
        unique_dates = df['date'].unique()[:10]
        realizations =[]
        
        for date in unique_dates:
            day_df = df[df['date'] == date]
            t_arrays =[]
            for i in range(8):
                ts = np.sort(day_df[day_df['event_id'] == i]['time'].values) - 34200.0
                jitter = np.random.uniform(1e-9, 1e-7, size=len(ts))
                t_arrays.append(np.sort(ts.astype(float) + jitter))
            realizations.append(t_arrays)

        em = HawkesEM(kernel_support=5.0, kernel_size=50, n_threads=1, verbose=False, max_iter=20)
        em.fit(realizations)
        
        t_values = np.linspace(0, 5.0, 50)
        total_fits = 64
        current_fit = 0

        for i in range(8):
            for j in range(8):
                y_values = em.get_kernel_values(i, j, t_values)
                alpha[i, j] = 0.0
                
                if np.isfinite(y_values).all():
                    try:
                        integral_norm = np.trapezoid(y_values, t_values)
                    except AttributeError:
                        integral_norm = np.trapz(y_values, t_values)
                        
                    if integral_norm >= 0.01:
                        try:
                            popt, _ = curve_fit(
                                power_law, 
                                t_values, 
                                y_values, 
                                p0=[0.05, 2.0, 1.2],
                                bounds=([0.0, 0.001, 1.001],[10.0, 100.0, 2.0]),
                                maxfev=10000
                            )
                            alpha[i, j] = popt[0]
                            gamma[i, j] = popt[1]
                            beta[i, j]  = popt[2]
                            M_hat[i, j] = popt[0] / (popt[1] * (popt[2] - 1))
                        except RuntimeError:
                            pass

                current_fit += 1

        eigenvalues = np.linalg.eigvals(M_hat)
        max_eig = np.max(np.real(eigenvalues))
        max_allowed_eig = 1.0 / max_f_Qt
        
        if max_eig >= max_allowed_eig:
            scale_factor = (max_allowed_eig - 0.01) / max_eig
            M_hat = M_hat * scale_factor
            alpha = alpha * scale_factor

    I = np.eye(8)
    mu_hat = np.maximum(0, np.dot((I - M_hat), lambda_hat))

    config['mu_base'] = mu_hat.tolist()
    config['alpha_matrix'] = alpha.tolist()
    config['gamma_matrix'] = gamma.tolist()
    config['beta_matrix'] = beta.tolist()

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"[{ticker}] successfully saved.\n")

if __name__ == "__main__":
    num_cores = min(mp.cpu_count(), len(TICKERS))
    print(f"Launching {num_cores} cores for calibration...")
    with mp.Pool(processes=num_cores) as pool:
        pool.map(calibrate_ticker, TICKERS)