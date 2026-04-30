import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Momentum_Algorithm import MAgent

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, "Plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def parse_time(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def load_and_prep_taq(quotes_file, trades_file):
    quotes = pd.read_csv(quotes_file)
    trades = pd.read_csv(trades_file)
    
    quotes['TIME_SEC'] = quotes['TIME_M'].apply(parse_time)
    trades['TIME_SEC'] = trades['TIME_M'].apply(parse_time)
    
    quotes = quotes[quotes['ASK'] > quotes['BID']].copy()
    quotes['EVENT_TYPE'] = 'QUOTE'
    quotes = quotes[['TIME_SEC', 'EVENT_TYPE', 'BID', 'ASK']]
    
    trades['EVENT_TYPE'] = 'TRADE'
    trades = trades[['TIME_SEC', 'EVENT_TYPE', 'PRICE', 'SIZE']]
    
    events = pd.concat([quotes, trades]).sort_values(by=['TIME_SEC', 'EVENT_TYPE']).reset_index(drop=True)
    events['BID'] = events['BID'].ffill()
    events['ASK'] = events['ASK'].ffill()
    events = events.dropna(subset=['BID', 'ASK']).copy()
    
    events['MID'] = (events['BID'] + events['ASK']) / 2.0
    events['SIDE'] = np.where(events['PRICE'] >= events['ASK'], 'BUY',
                     np.where(events['PRICE'] <= events['BID'], 'SELL', 'UNKNOWN'))
    return events

def run_single_day(events_df, date_str):
    agent = MAgent(lot_size=100, Q_bar=50)
    MAX_SHARES = agent.Q_bar * agent.lot_size
    
    inventory = 0
    cash = 0.0
    
    pnl_history, time_history, inv_history = [], [], []
    current_bid, current_ask = 0.0, 0.0
    active_z_bid, active_z_ask = 0, 0
    
    MARKET_OPEN_SEC = 9.5 * 3600   
    WARMUP_SEC = 900
    MARKET_CLOSE_SEC = 16.0 * 3600 
    
    for row in events_df.itertuples():
        t_sec = row.TIME_SEC
        time_to_close = MARKET_CLOSE_SEC - t_sec
        
        # 0. WARMUP
        if t_sec < MARKET_OPEN_SEC + WARMUP_SEC:
            if row.EVENT_TYPE == 'TRADE' and row.SIDE != 'UNKNOWN':
                agent.update_alpha(t_sec, row.SIDE, row.SIZE)
            continue
        
        # 1. EOD LIQUIDATION
        if time_to_close < 10.0:
            if inventory > 0: cash += inventory * current_bid
            elif inventory < 0: cash -= abs(inventory) * current_ask
            inventory = 0
            
            pnl_history.append(cash)
            time_history.append(t_sec)
            inv_history.append(inventory)
            break  
            
        # 2. QUOTES & AGENT ACTION
        if row.EVENT_TYPE == 'QUOTE':
            if row.BID != current_bid or row.ASK != current_ask:
                active_z_bid, active_z_ask = 0, 0
                
            current_bid, current_ask = row.BID, row.ASK
            
            mo, z_ask, z_bid = agent.get_action(t_sec, inventory, time_to_close, current_bid, current_ask)
            
            if mo != 0:
                side = 'BUY' if mo > 0 else 'SELL'
                size = abs(mo) * 100
                
                if side == 'BUY':
                    cash -= size * (current_ask + 0.002) # Pay spread + fee
                    inventory += size
                elif side == 'SELL':
                    cash += size * (current_bid - 0.002)
                    inventory -= size
                    
                active_z_bid, active_z_ask = 0, 0
            else:
                active_z_ask = z_ask * 100
                active_z_bid = z_bid * 100
                
        elif row.EVENT_TYPE == 'TRADE':
            trade_price, trade_size, trade_side = row.PRICE, row.SIZE, row.SIDE
            
            if trade_side != 'UNKNOWN':
                agent.update_alpha(t_sec, trade_side, trade_size)
            
            if trade_side == 'BUY' and trade_price >= current_ask and active_z_ask > 0:
                # Guaranteed minimum 1 lot fill if hit
                filled_qty = min(active_z_ask, max(100, int(trade_size * 0.25)))
                filled_qty = min(filled_qty, MAX_SHARES + inventory) # Capacity check
                if filled_qty > 0:
                    inventory -= filled_qty
                    cash += filled_qty * current_ask
                    active_z_ask -= filled_qty
                
            elif trade_side == 'SELL' and trade_price <= current_bid and active_z_bid > 0:
                filled_qty = min(active_z_bid, max(100, int(trade_size * 0.25)))
                filled_qty = min(filled_qty, MAX_SHARES - inventory)
                if filled_qty > 0:
                    inventory += filled_qty
                    cash -= filled_qty * current_bid
                    active_z_bid -= filled_qty

        # 4. LOG METRICS
        if len(time_history) == 0 or (t_sec - time_history[-1]) >= 60.0:
            mid = (current_bid + current_ask) / 2.0
            unrealized = inventory * mid if mid > 0 else 0
            pnl_history.append(cash + unrealized)
            time_history.append(t_sec)
            inv_history.append(inventory)

    daily_total_pnl = pnl_history[-1] if pnl_history else 0
    minute_returns = pd.Series(pnl_history).diff().dropna().tolist()
    print(f"[{date_str}] Day PnL: ${daily_total_pnl:.2f} | Final Inv: {inventory}")
    return daily_total_pnl, minute_returns, time_history, pnl_history, inv_history

def run_multi_day_backtest(ticker, base_dir):
    quotes_dir = os.path.join(base_dir, ticker, f"{ticker}_Quotes")
    trades_dir = os.path.join(base_dir, ticker, f"{ticker}_Trades")
    
    quote_files = sorted(glob.glob(os.path.join(quotes_dir, "*.csv")))
    cumulative_pnl = 0.0
    daily_pnls, all_minute_returns = [],[]
    
    print(f"=== Starting Robust HJB Backtest for {ticker} ===")
    
    for q_file in quote_files:
        date_str = os.path.basename(q_file).split('_')[1] 
        t_file = os.path.join(trades_dir, f"{ticker}_{date_str}_trades.csv")
        
        if not os.path.exists(t_file): continue
            
        events_df = load_and_prep_taq(q_file, t_file)
        
        day_pnl, day_returns, t_hist, pnl_hist, inv_hist = run_single_day(events_df, date_str)

        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.plot(t_hist, pnl_hist, color='green', linewidth=2)
        plt.title(f'{ticker} - {date_str} - PnL')
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 1, 2)
        plt.plot(t_hist, inv_hist, color='orange', linewidth=2)
        plt.title('Inventory')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{ticker}_{date_str}.png"))
        
        plt.show() 
        
        cumulative_pnl += day_pnl
        daily_pnls.append(cumulative_pnl)
        all_minute_returns.extend(day_returns)
        
    returns_array = np.array(all_minute_returns)
    sharpe = np.sqrt(390 * 252) * (returns_array.mean() / returns_array.std()) if len(returns_array) > 0 else 0

    print("\n" + "="*40)
    print(f"  FINAL {ticker} RESULTS  ")
    print("="*40)
    print(f"Total Cumulative PnL : ${cumulative_pnl:.2f}")
    print(f"Overall Ann. Sharpe  : {sharpe:.2f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(daily_pnls) + 1), daily_pnls, color='blue', marker='o')
    plt.title(f'{ticker} Cumulative PnL')
    plt.xlabel('Trading Days')
    plt.ylabel('PnL ($)')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    BASE_DATA_DIR = "Extracted_Market_Data"
    TICKER = "IBM"
    run_multi_day_backtest(TICKER, BASE_DATA_DIR)