import os
import io
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Liquidity_Momentum_Algorithm import LMAgent

def parse_time(time_str):
    """Converts time string 'HH:MM:SS.ns' to seconds from midnight."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def load_and_prep_taq(quotes_file, trades_file):
    quotes = pd.read_csv(quotes_file)
    trades = pd.read_csv(trades_file)
    
    quotes['TIME_SEC'] = quotes['TIME_M'].apply(parse_time)
    trades['TIME_SEC'] = trades['TIME_M'].apply(parse_time)
    
    quotes = quotes[quotes['ASK'] > quotes['BID']].copy()
    quotes['EVENT_TYPE'] = 'QUOTE'
    quotes = quotes[['TIME_SEC', 'EVENT_TYPE', 'BID', 'ASK', 'BIDSIZ', 'ASKSIZ']]
    
    trades['EVENT_TYPE'] = 'TRADE'
    trades = trades[['TIME_SEC', 'EVENT_TYPE', 'PRICE', 'SIZE']]
    
    events = pd.concat([quotes, trades]).sort_values(by=['TIME_SEC', 'EVENT_TYPE']).reset_index(drop=True)
    events['BID'] = events['BID'].ffill()
    events['ASK'] = events['ASK'].ffill()
    events['BIDSIZ'] = events['BIDSIZ'].ffill()
    events['ASKSIZ'] = events['ASKSIZ'].ffill()
    events = events.dropna(subset=['BID', 'ASK']).copy()
    
    events['MID'] = (events['BID'] + events['ASK']) / 2.0
    events['SIDE'] = np.where(events['PRICE'] > events['MID'], 'BUY',
                     np.where(events['PRICE'] < events['MID'], 'SELL', 'UNKNOWN'))
    
    print(f"Data prepped! Total events: {len(events)}")
    return events

def run_backtest(events_df):
    agent = LMAgent(lot_size=100, Q_bar=50)
    MAX_SHARES = agent.Q_bar * agent.lot_size
    
    inventory = 0
    cash = 0.0
    
    pnl_history = []
    time_history = []
    inv_history =[]
    
    current_bid = 0.0
    current_ask = 0.0
    
    current_bidsiz_shares = 100.0 
    current_asksiz_shares = 100.0
    
    active_z_bid = 0
    active_z_ask = 0
    
    MARKET_OPEN_SEC = 9.5 * 3600   # 9:30 AM
    WARMUP_SEC = 300               # 5 minutes
    MARKET_CLOSE_SEC = 16.0 * 3600 # 4:00 PM
    
    print("\nStarting Simulation Loop...")
    
    for row in events_df.itertuples():
        t_sec = row.TIME_SEC
        time_to_close = MARKET_CLOSE_SEC - t_sec
        
        if row.EVENT_TYPE == 'QUOTE':
            if row.BID != current_bid or row.ASK != current_ask:
                active_z_bid = 0
                active_z_ask = 0
                
            current_bid = row.BID
            current_ask = row.ASK
            current_bidsiz_shares = max(100.0, row.BIDSIZ * 100.0)
            current_asksiz_shares = max(100.0, row.ASKSIZ * 100.0)
            
            # Only ask the agent for actions if warmup is over and EOD hasn't hit
            if (t_sec >= MARKET_OPEN_SEC + WARMUP_SEC) and (time_to_close >= 10.0):
                mo, z_ask, z_bid = agent.get_action(t_sec, inventory, time_to_close, current_bidsiz_shares, current_asksiz_shares)
                
                if mo != 0:
                    side = 'BUY' if mo > 0 else 'SELL'
                    size = abs(mo) * 100
                    
                    if side == 'BUY' and inventory < 0:
                        size = min(size, abs(inventory))
                        cash -= size * (current_ask + 0.002) # Add fee
                        inventory += size
                        print(f"[{t_sec:.2f}] MO BUY: {size} shares @ {current_ask:.2f} | Inv: {inventory}")
                    elif side == 'SELL' and inventory > 0:
                        size = min(size, inventory)
                        cash += size * (current_bid - 0.002) # Subtract fee
                        inventory -= size
                        print(f"[{t_sec:.2f}] MO SELL: {size} shares @ {current_bid:.2f} | Inv: {inventory}")
                        
                    active_z_bid, active_z_ask = 0, 0
                else:
                    target_ask = z_ask * 100
                    target_bid = z_bid * 100
                    
                    if target_ask > active_z_ask: active_z_ask = target_ask
                    elif target_ask == 0: active_z_ask = 0
                    
                    if target_bid > active_z_bid: active_z_bid = target_bid
                    elif target_bid == 0: active_z_bid = 0

        if t_sec < MARKET_OPEN_SEC + WARMUP_SEC:
            if row.EVENT_TYPE == 'TRADE' and row.SIDE != 'UNKNOWN':
                # FIX: Add the prevailing liquidity to the warm-up alpha update
                prevailing_liquidity = current_asksiz_shares if row.SIDE == 'BUY' else current_bidsiz_shares
                agent.update_alpha(t_sec, row.SIDE, row.SIZE, prevailing_liquidity / agent.lot_size)
            continue 
            
        if time_to_close < 10.0:
            if inventory > 0:
                cash += inventory * current_bid
                print(f"[{t_sec:.2f}] EOD: Sold {inventory} shares at {current_bid:.2f}")
            elif inventory < 0:
                cash -= abs(inventory) * current_ask
                print(f"[{t_sec:.2f}] EOD: Bought {abs(inventory)} shares at {current_ask:.2f}")
            
            inventory = 0
            
            pnl_history.append(cash)
            time_history.append(t_sec)
            inv_history.append(inventory)
            
            break
                
        elif row.EVENT_TYPE == 'TRADE':
            trade_price = row.PRICE
            trade_size = row.SIZE
            trade_side = row.SIDE
            
            if trade_side != 'UNKNOWN':
                prevailing_liquidity = current_asksiz_shares if trade_side == 'BUY' else current_bidsiz_shares
                agent.update_alpha(t_sec, trade_side, trade_size, prevailing_liquidity / agent.lot_size)
            
            max_can_buy = MAX_SHARES - inventory
            max_can_sell = MAX_SHARES + inventory
            
            if trade_side == 'BUY' and trade_price >= current_ask and active_z_ask > 0:
                consumption_ratio = min(1.0, trade_size / current_asksiz_shares)
                filled_qty = active_z_ask * consumption_ratio 
                
                if filled_qty >= 1:
                    filled_qty = int(filled_qty)
                    filled_qty = min(filled_qty, max_can_sell)
                    inventory -= filled_qty
                    cash += filled_qty * current_ask
                    active_z_ask -= filled_qty
                
            elif trade_side == 'SELL' and trade_price <= current_bid and active_z_bid > 0:
                consumption_ratio = min(1.0, trade_size / current_bidsiz_shares)
                filled_qty = active_z_bid * consumption_ratio
                
                if filled_qty >= 1:
                    filled_qty = int(filled_qty)
                    filled_qty = min(filled_qty, max_can_buy)
                    inventory += filled_qty
                    cash -= filled_qty * current_bid
                    active_z_bid -= filled_qty

        if len(time_history) == 0 or (t_sec - time_history[-1]) >= 60.0:
            mid = (current_bid + current_ask) / 2.0
            unrealized = inventory * mid if mid > 0 else 0
            total_pnl = cash + unrealized
            
            pnl_history.append(total_pnl)
            time_history.append(t_sec)
            inv_history.append(inventory)

    pnl_series = pd.Series(pnl_history)
    minute_returns = pnl_series.diff().dropna()
    total_pnl = pnl_series.iloc[-1] if not pnl_series.empty else 0
    
    sharpe = 0.0
    if not minute_returns.empty and minute_returns.std() > 0:
        sharpe = np.sqrt(390 * 252) * (minute_returns.mean() / minute_returns.std())

    print("\n" + "="*30)
    print("      BACKTEST RESULTS      ")
    print("="*30)
    print(f"Final Inventory : {inventory} shares")
    print(f"Total PnL       : ${total_pnl:.2f}")
    print(f"Ann. Sharpe     : {sharpe:.2f}")
    print("="*30)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_history, pnl_history, color='green', label='Cumulative PnL ($)')
    plt.title('4D HJB Market Maker Performance')
    plt.ylabel('PnL ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time_history, inv_history, color='orange', label='Inventory (Shares)')
    plt.ylabel('Inventory')
    plt.xlabel('Time (Seconds from Midnight)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    q_file = "Extracted_Market_Data\KO\KO_Quotes\KO_20251002_quotes.csv"
    t_file = "Extracted_Market_Data\KO\KO_Trades\KO_20251002_trades.csv"
    
    events_df = load_and_prep_taq(q_file, t_file)
    run_backtest(events_df)