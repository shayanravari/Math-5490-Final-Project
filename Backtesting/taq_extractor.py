import pandas as pd
import os
import glob

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TAQ_DIR = os.path.join(ROOT_DIR, "TAQ_Market_Data")
OUT_DIR = os.path.join(ROOT_DIR, "Extracted_Market_Data")

TRADE_FILE = os.path.join(TAQ_DIR, "ychp7yleedp0qebd.csv.gz")
QUOTE_FILE = os.path.join(TAQ_DIR, "tjz3c3cg4tgptk73.csv.gz")

TICKERS = ["KO", "TGT"]
CHUNK_SIZE = 1_000_000     

def setup_directories():
    for ticker in TICKERS:
        ticker_base = os.path.join(OUT_DIR, ticker)
        
        t_dir = os.path.join(ticker_base, f"{ticker}_Trades")
        q_dir = os.path.join(ticker_base, f"{ticker}_Quotes")
        
        os.makedirs(t_dir, exist_ok=True)
        os.makedirs(q_dir, exist_ok=True)
        
        for f in glob.glob(os.path.join(t_dir, "*.csv")): os.remove(f)
        for f in glob.glob(os.path.join(q_dir, "*.csv")): os.remove(f)

def extract_and_split(file_path, file_type):
    print(f"\nExtracting {file_type.upper()} for {TICKERS}")
    if not os.path.exists(file_path):
        print(f"Cannot find {file_path}")
        return

    try:
        chunk_iter = pd.read_csv(file_path, compression='gzip', chunksize=CHUNK_SIZE, low_memory=False)
        for i, chunk in enumerate(chunk_iter):
            ticker_col = 'SYM_ROOT' if 'SYM_ROOT' in chunk.columns else 'Symbol'
            date_col = 'DATE' if 'DATE' in chunk.columns else 'Date'
            time_col = 'TIME_M' if 'TIME_M' in chunk.columns else 'Time'
            
            chunk[ticker_col] = chunk[ticker_col].astype(str).str.strip()
            filtered_chunk = chunk[chunk[ticker_col].isin(TICKERS)].copy()
            
            if not filtered_chunk.empty:
                time_parts = filtered_chunk[time_col].astype(str).str.split(':', expand=True)
                if time_parts.shape[1] >= 3: 
                    total_seconds = (time_parts[0].astype(float) * 3600 + time_parts[1].astype(float) * 60 + time_parts[2].astype(float))
                    filtered_chunk = filtered_chunk[(total_seconds >= 34200) & (total_seconds <= 57600)]
                
                if not filtered_chunk.empty:
                    for (date, ticker), group_df in filtered_chunk.groupby([date_col, ticker_col]):
                        safe_date = str(date).replace('/', '').replace('-', '')
                        folder_type = f"{ticker}_Trades" if file_type == "trades" else f"{ticker}_Quotes"
                        output_name = os.path.join(OUT_DIR, ticker, folder_type, f"{ticker}_{safe_date}_{file_type}.csv")
                        write_header = not os.path.exists(output_name)
                        with open(output_name, 'a', newline='') as f:
                            group_df.to_csv(f, header=write_header, index=False)
                            
            print(f"Processed chunk {i+1} ({(i+1) * CHUNK_SIZE:,} rows)")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    setup_directories()
    extract_and_split(TRADE_FILE, "trades")
    extract_and_split(QUOTE_FILE, "quotes")