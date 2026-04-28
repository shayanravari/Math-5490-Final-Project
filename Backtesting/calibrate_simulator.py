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

    sizes = df['size'].values
    total_orders = len(sizes)
    round_nums =[1, 10, 50, 100, 200, 500]
    round_counts = {rn: np.sum(sizes == rn) for rn in round_nums}
    total_round = sum(round_counts.values())
    
    config['order_size_model'] = {
        'round_numbers': round_nums,
        'round_probs':[float(round_counts[rn]/total_round) for rn in round_nums] if total_round > 0 else[0.1,0.3,0.2,0.2,0.1,0.1],
        'geometric_p': float(min(1.0, 1.0 / df[~df['size'].isin(round_nums)]['size'].mean())) if not df[~df['size'].isin(round_nums)].empty else 0.05,
        'p_round': float(total_round / total_orders) if total_orders > 0 else 0.75
    }

    num_days = max(1, df['date'].nunique())
    total_sim_seconds = num_days * (6.5 * 3600)
    
    lambda_hat = np.zeros(12)
    event_counts = df['event_id'].value_counts().to_dict()
    for i in range(12):
        lambda_hat[i] = event_counts.get(i, 0) / total_sim_seconds

    df['bin'] = ((df['time'] - 34200) // 1800).astype(int)
    bin_counts = df[df['bin'] <= 12]['bin'].value_counts().sort_index()
    avg_per_bin = bin_counts.mean()
    u_shape = {str(int(b)): round(float(c / avg_per_bin), 2) for b, c in bin_counts.items()}
    config['u_shape_bins'] = u_shape
    
    max_f_Qt = max(u_shape.values()) if u_shape else 1.0

    alpha = np.zeros((12, 12))
    gamma = np.ones((12, 12)) 
    beta = np.full((12, 12), 1.5) 
    M_hat = np.zeros((12, 12)) 
    
    if TICK_AVAILABLE:
        unique_dates = df['date'].unique()[:3]
        realizations =[]
        
        for date in unique_dates:
            day_df = df[df['date'] == date]
            t_arrays =[]
            for i in range(12):
                ts = np.sort(day_df[day_df['event_id'] == i]['time'].values) - 34200.0
                jitter = np.random.uniform(1e-9, 1e-7, size=len(ts))
                ts_jittered = np.sort(ts.astype(float) + jitter)
                
                t_arrays.append(ts_jittered)
                
            realizations.append(t_arrays)

        em = HawkesEM(kernel_support=5.0, kernel_size=50, n_threads=1, verbose=True, max_iter=20)
        
        print(f"\nSTARTING EM SOLVER")
        em.fit(realizations)
        print(f"EM SOLVER FINISHED\n")
        
        t_values = np.linspace(0, 5.0, 50)

        print("Extracting and fitting 144 Power-Law Curves.")
        total_fits = 144
        current_fit = 0

        for i in range(12):
            for j in range(12):
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
                                p0=[0.1, 1.0, 1.5], 
                                bounds=([0, 0.001, 1.001],[10.0, 100.0, 10.0]),
                                maxfev=5000
                            )
                            alpha[i, j] = popt[0]
                            gamma[i, j] = popt[1]
                            beta[i, j]  = popt[2]
                            M_hat[i, j] = popt[0] / (popt[1] * (popt[2] - 1))
                        except RuntimeError:
                            pass

                current_fit += 1
                if current_fit % 12 == 0:
                    print(f"        -> Fit Progress: {current_fit}/{total_fits} curves processed.")

        eigenvalues = np.linalg.eigvals(M_hat)
        max_eig = np.max(np.real(eigenvalues))
        max_allowed_eig = 1.0 / max_f_Qt
        
        print(f"\nCalculated Max Eigenvalue: {max_eig:.4f} | Limit (Eq 21): < {max_allowed_eig:.4f}")

        if max_eig >= max_allowed_eig:
            print(f"Eigenvalue exceeds bound.")
            scale_factor = (max_allowed_eig - 0.01) / max_eig
            M_hat = M_hat * scale_factor
            alpha = alpha * scale_factor

    print("Recalibrating exogenous intensities.")
    I = np.eye(12)
    mu_hat = np.dot((I - M_hat), lambda_hat)
    mu_hat = np.maximum(0, mu_hat)

    config['mu_base'] = mu_hat.tolist()
    config['alpha_matrix'] = alpha.tolist()
    config['gamma_matrix'] = gamma.tolist()
    config['beta_matrix'] = beta.tolist()

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"[{ticker}] successfully saved.\n")

if __name__ == "__main__":
    num_cores = min(mp.cpu_count(), len(TICKERS))
    with mp.Pool(processes=num_cores) as pool:
        pool.map(calibrate_ticker, TICKERS)