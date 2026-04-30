import pandas as pd
import os
import glob
import multiprocessing as mp

# --- ROBUST FOLDER PATHING ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TAQ_DIR = os.path.join(ROOT_DIR, "TAQ_Market_Data")
OUT_DIR = os.path.join(ROOT_DIR, "Extracted_Market_Data")

TRADE_FILE = os.path.join(TAQ_DIR, "ychp7yleedp0qebd.csv.gz")
QUOTE_FILE = os.path.join(TAQ_DIR, "tjz3c3cg4tgptk73.csv.gz")

TICKERS = ["IBM", "KO", "TGT"]
CHUNK_SIZE = 1_000_000     

TRADE_COLS =['DATE', 'Date', 'TIME_M', 'Time', 'EX', 'SYM_ROOT', 'Symbol', 'SIZE', 'PRICE', 'TR_SCOND']
QUOTE_COLS =['DATE', 'Date', 'TIME_M', 'Time', 'EX', 'SYM_ROOT', 'Symbol', 'BID', 'BIDSIZ', 'ASK', 'ASKSIZ']

def setup_directories():
    for ticker in TICKERS:
        ticker_base = os.path.join(OUT_DIR, ticker)
        t_dir = os.path.join(ticker_base, f"{ticker}_Trades")
        q_dir = os.path.join(ticker_base, f"{ticker}_Quotes")
        
        os.makedirs(t_dir, exist_ok=True)
        os.makedirs(q_dir, exist_ok=True)
        
        for f in glob.glob(os.path.join(t_dir, "*.csv")): os.remove(f)
        for f in glob.glob(os.path.join(q_dir, "*.csv")): os.remove(f)

def extract_and_split(args):
    file_path, file_type = args
    
    if not os.path.exists(file_path):
        print(f"ERROR: Cannot find {file_path}")
        return

    use_cols = TRADE_COLS if file_type == "trades" else QUOTE_COLS

    try:
        chunk_iter = pd.read_csv(
            file_path, 
            compression='gzip', 
            chunksize=CHUNK_SIZE, 
            usecols=lambda c: c in use_cols,
            low_memory=False
        )
        
        for i, chunk in enumerate(chunk_iter):
            if 'EX' in chunk.columns:
                chunk = chunk[chunk['EX'] == 'T'] # Nasdaq
            if chunk.empty: continue
            
            ticker_col = 'SYM_ROOT' if 'SYM_ROOT' in chunk.columns else 'Symbol'
            chunk[ticker_col] = chunk[ticker_col].astype(str).str.strip()
            chunk = chunk[chunk[ticker_col].isin(TICKERS)].copy()
            if chunk.empty: continue
            
            # Time Filtering
            time_col = 'TIME_M' if 'TIME_M' in chunk.columns else 'Time'
            time_parts = chunk[time_col].astype(str).str.split(':', expand=True)
            
            if time_parts.shape[1] >= 3: 
                total_seconds = (time_parts[0].astype(float) * 3600 + 
                                 time_parts[1].astype(float) * 60 + 
                                 time_parts[2].astype(float))
                chunk = chunk[(total_seconds >= 34200) & (total_seconds <= 57600)]
            
            if chunk.empty: continue
            
            # Save Data
            date_col = 'DATE' if 'DATE' in chunk.columns else 'Date'
            
            for (date, ticker), group_df in chunk.groupby([date_col, ticker_col]):
                group_df = group_df.copy()
                safe_date = str(date).replace('/', '').replace('-', '')
                
                if file_type == "quotes":
                    if safe_date < '20251101':
                        group_df['BIDSIZ'] = group_df['BIDSIZ'] * 100
                        group_df['ASKSIZ'] = group_df['ASKSIZ'] * 100
                
                folder_type = f"{ticker}_Trades" if file_type == "trades" else f"{ticker}_Quotes"
                output_name = os.path.join(OUT_DIR, ticker, folder_type, f"{ticker}_{safe_date}_{file_type}.csv")
                
                write_header = not os.path.exists(output_name)
                with open(output_name, 'a', newline='') as f:
                    group_df.to_csv(f, header=write_header, index=False)
                    
            print(f"[{file_type.upper()}] Processed chunk {i+1} ({(i+1) * CHUNK_SIZE:,} rows parsed)")
            
    except Exception as e:
        print(f"Error reading {file_type} file: {e}")

if __name__ == "__main__":
    setup_directories()
    
    tasks =[
        (TRADE_FILE, "trades"),
        (QUOTE_FILE, "quotes")
    ]
    
    with mp.Pool(processes=2) as pool:
        pool.map(extract_and_split, tasks)
        