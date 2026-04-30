import pandas as pd
import numpy as np
import os
import glob
import multiprocessing as mp

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(ROOT_DIR, "Extracted_Market_Data")

TICKERS = ["IBM", "KO", "TGT"] 

E = {
    'LO_ask0': 0, 'CO_ask0': 1, 'MO_ask0': 2, 'LO_ask-1': 3,
    'LO_bid+1': 4, 'LO_bid0': 5, 'CO_bid0': 6, 'MO_bid0': 7
}

def time_to_seconds(t_str):
    try:
        h, m, s = str(t_str).split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    except:
        return 0.0
    
def process_single_day(args):
    ticker, date_str, tf, qf = args
    
    try:
        trades = pd.read_csv(tf, usecols=['TIME_M', 'PRICE', 'SIZE'])
        trades['TYPE'] = 1 
        
        quotes = pd.read_csv(qf, usecols=['TIME_M', 'BID', 'BIDSIZ', 'ASK', 'ASKSIZ'])
        quotes['TYPE'] = 0
    except Exception as e:
        print(f"Error reading files for {date_str}: {e}")
        return pd.DataFrame()

    trades['time_sec'] = trades['TIME_M'].apply(time_to_seconds)
    quotes['time_sec'] = quotes['TIME_M'].apply(time_to_seconds)
    
    df = pd.concat([trades, quotes]).sort_values(by=['time_sec', 'TYPE'], ascending=[True, False]).reset_index(drop=True)

    df['PRICE'] = df['PRICE'].fillna(0.0)
    df['SIZE'] = df['SIZE'].fillna(0.0)
    df['BID'] = df['BID'].fillna(0.0)
    df['BIDSIZ'] = df['BIDSIZ'].fillna(0.0)
    df['ASK'] = df['ASK'].fillna(0.0)
    df['ASKSIZ'] = df['ASKSIZ'].fillna(0.0)

    t_arr = df['time_sec'].values
    type_arr = df['TYPE'].values
    price_arr = df['PRICE'].values
    size_arr = df['SIZE'].values
    bid_arr = df['BID'].values
    bidsiz_arr = df['BIDSIZ'].values
    ask_arr = df['ASK'].values
    asksiz_arr = df['ASKSIZ'].values

    curr_bid, curr_bidsiz, curr_ask, curr_asksiz = 0.0, 0, float('inf'), 0
    pending_mo_bid_size, pending_mo_ask_size = 0, 0
    
    daily_events =[]
    
    for i in range(len(t_arr)):
        t = t_arr[i]
        row_type = type_arr[i]
        
        if row_type == 1: 
            price = price_arr[i]
            size = size_arr[i]
            if price >= curr_ask:
                daily_events.append({'date': date_str, 'time': t, 'event_id': E['MO_ask0'], 'size': size})
                pending_mo_ask_size += size
            elif price <= curr_bid:
                daily_events.append({'date': date_str, 'time': t, 'event_id': E['MO_bid0'], 'size': size})
                pending_mo_bid_size += size
                
        elif row_type == 0:
            new_bid = bid_arr[i]
            new_bidsiz = bidsiz_arr[i]
            new_ask = ask_arr[i]
            new_asksiz = asksiz_arr[i]
            
            if new_bid == 0.0 or new_ask == 0.0 or new_bid >= new_ask: 
                continue

            # BID SIDE
            if new_bid > curr_bid:
                daily_events.append({'date': date_str, 'time': t, 'event_id': E['LO_bid+1'], 'size': new_bidsiz})
                pending_mo_bid_size = 0
            elif new_bid == curr_bid:
                if new_bidsiz > curr_bidsiz:
                    daily_events.append({'date': date_str, 'time': t, 'event_id': E['LO_bid0'], 'size': new_bidsiz - curr_bidsiz})
                elif new_bidsiz < curr_bidsiz:
                    drop_size = curr_bidsiz - new_bidsiz
                    deduct = min(drop_size, pending_mo_bid_size)
                    drop_size -= deduct
                    pending_mo_bid_size -= deduct
                    if drop_size > 0: 
                        daily_events.append({'date': date_str, 'time': t, 'event_id': E['CO_bid0'], 'size': drop_size})
            elif new_bid < curr_bid:
                if pending_mo_bid_size < curr_bidsiz:
                    daily_events.append({'date': date_str, 'time': t, 'event_id': E['CO_bid0'], 'size': curr_bidsiz - pending_mo_bid_size})
                pending_mo_bid_size = 0

            # ASK SIDE
            if new_ask < curr_ask:
                daily_events.append({'date': date_str, 'time': t, 'event_id': E['LO_ask-1'], 'size': new_asksiz})
                pending_mo_ask_size = 0
            elif new_ask == curr_ask:
                if new_asksiz > curr_asksiz:
                    daily_events.append({'date': date_str, 'time': t, 'event_id': E['LO_ask0'], 'size': new_asksiz - curr_asksiz})
                elif new_asksiz < curr_asksiz:
                    drop_size = curr_asksiz - new_asksiz
                    deduct = min(drop_size, pending_mo_ask_size)
                    drop_size -= deduct
                    pending_mo_ask_size -= deduct
                    if drop_size > 0: 
                        daily_events.append({'date': date_str, 'time': t, 'event_id': E['CO_ask0'], 'size': drop_size})
            elif new_ask > curr_ask:
                if pending_mo_ask_size < curr_asksiz:
                    daily_events.append({'date': date_str, 'time': t, 'event_id': E['CO_ask0'], 'size': curr_asksiz - pending_mo_ask_size})
                pending_mo_ask_size = 0

            curr_bid, curr_bidsiz, curr_ask, curr_asksiz = new_bid, new_bidsiz, new_ask, new_asksiz

    print(f" -> [{ticker}] Finished {date_str} (Extracted {len(daily_events):,} events)", flush=True)
    return pd.DataFrame(daily_events)

def parse_ticker(ticker):
    base_dir = os.path.join(OUT_DIR, ticker)
    trade_dir = os.path.join(base_dir, f"{ticker}_Trades")
    quote_dir = os.path.join(base_dir, f"{ticker}_Quotes")
    out_file = os.path.join(base_dir, f"{ticker}_Hawkes_Events.csv")
    
    print(f"\n{'='*40}\nPARSING TICKER: {ticker}\n{'='*40}")
    
    trade_files = sorted(glob.glob(os.path.join(trade_dir, "*_trades.csv")))

    args_list =[]
    for tf in trade_files:
        date_str = os.path.basename(tf).split('_')[1]
        qf = os.path.join(quote_dir, f"{ticker}_{date_str}_quotes.csv")
        
        if not os.path.exists(qf): 
            print(f"Missing quote file for {date_str}.")
            continue
            
        args_list.append((ticker, date_str, tf, qf))

    num_cores = mp.cpu_count()
    
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(process_single_day, args_list)

    valid_results = [df for df in results if not df.empty]
    
    if valid_results:
        final_df = pd.concat(valid_results, ignore_index=True)
        final_df['size'] = final_df['size'].astype(int)
        final_df.to_csv(out_file, index=False)
        print(f"\n=> Saved {len(final_df):,} total events for {ticker} to {out_file}!\n")
    else:
        print(f"\n=> No events parsed for {ticker}.")

if __name__ == "__main__":
    for t in TICKERS:
        parse_ticker(t)