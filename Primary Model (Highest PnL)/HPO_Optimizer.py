import numpy as np
from Main_Algorithm import run_multi_day_backtest 

if __name__ == "__main__":
    quotes_directory = "Extracted_Market_Data/KO/KO_Quotes"
    trades_directory = "Extracted_Market_Data/KO/KO_Trades"
    
    # Use just the first 3 or 5 days of October as your "Train Set"
    dates_to_train =["20251001", "20251002", "20251003"]
    
    print("Monte Carlo Hyperparameter Optimization")
    
    best_pnl = -9999.0
    best_params = {}
    
    NUM_ITERATIONS = 50  
    
    for i in range(NUM_ITERATIONS):
        test_phi = 10 ** np.random.uniform(-5.0, -3.0)       
        test_psi = 10 ** np.random.uniform(-2.5, 0.0)        
        test_alpha_scale = 10 ** np.random.uniform(-5.0, -3.0) 
        test_max_clip = int(np.random.uniform(5, 25))        
        
        print(f"\n[ITERATION {i+1}/{NUM_ITERATIONS}] Testing Configuration:")
        print(f"Phi: {test_phi:.6f} | Psi: {test_psi:.4f} | Alpha_Scale: {test_alpha_scale:.6f} | Max_Clip: {test_max_clip}")
        
        try:
            # Run Backtester
            total_pnl, sharpe, max_dd = run_multi_day_backtest(
                dates_to_train, quotes_directory, trades_directory, 
                ticker="KO", 
                phi_val=test_phi, psi_val=test_psi, alpha_val=test_alpha_scale, clip_val=test_max_clip,
                plot_results=False
            )
            
            # Objective Function
            score = total_pnl
            if max_dd > 1500.0:
                score -= (max_dd - 1500.0) 
                
            print(f"-> Result: PnL: ${total_pnl:.2f} | Sharpe: {sharpe:.2f} | Max DD: ${max_dd:.2f} | Score: {score:.2f}")
            
            if score > best_pnl:
                best_pnl = score
                best_params = {
                    'phi': test_phi, 'psi': test_psi, 'alpha_scale': test_alpha_scale, 'max_clip': test_max_clip,
                    'pnl': total_pnl, 'sharpe': sharpe, 'max_dd': max_dd
                }
                print("New Best Configuration")
                
        except Exception as e:
            print(f"Configuration failed: {e}")

    print("Optimized Hyperparameters:")
    print(f"phi={best_params['phi']:.6f}, psi={best_params['psi']:.4f}, alpha_scale={best_params['alpha_scale']:.6f}, max_clip_lots={best_params['max_clip']}")
    print("--------------------------------------------------------")
    print(f"Expected PnL on Train Set : ${best_params['pnl']:.2f}")
    print(f"Expected Sharpe           : {best_params['sharpe']:.2f}")
    print(f"Expected Max DD           : ${best_params['max_dd']:.2f}")