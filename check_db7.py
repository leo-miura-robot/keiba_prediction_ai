import sqlite3
import pandas as pd

db_old = r'D:\keiba\new_jra_2006-2015_fixed_20260616_015045\keiba.db'
db_new = r'D:\keiba\new_jra_2016-2026_fixed\keiba.db'

def extract_db_stats(path, name):
    conn = sqlite3.connect(path)
    print(f'\n=== JRDB Stats: {name} ===')
    
    # Check RA
    ra = pd.read_sql('SELECT SyussoTosu FROM NL_RA LIMIT 5000', conn)
    print(f'NL_RA SyussoTosu: {pd.to_numeric(ra["SyussoTosu"], errors="coerce").dropna().min()} ~ {pd.to_numeric(ra["SyussoTosu"], errors="coerce").dropna().max()}')

    # Check HR for Payouts
    hr = pd.read_sql('SELECT FukuPay, FukuPay2, FukuPay3 FROM NL_HR LIMIT 5000', conn)
    pays = []
    for c in ['FukuPay', 'FukuPay2', 'FukuPay3']:
        pays.extend(pd.to_numeric(hr[c], errors='coerce').dropna().tolist())
    pays = [p for p in pays if p > 0]
    if pays:
        print(f'NL_HR Fuku Payouts: min={min(pays)}, max={max(pays)}')
        
    # Check O1 for Odds
    o1 = pd.read_sql('SELECT FukuOddsLow, FukuOddsHigh FROM NL_O1 LIMIT 5000', conn)
    omin = pd.to_numeric(o1["FukuOddsLow"], errors="coerce").dropna()
    if len(omin) > 0:
        print(f'NL_O1 FukuOddsLow: min={omin.min()}, max={omin.max()}')
        
    # Check SE KettoNum length
    se = pd.read_sql('SELECT KettoNum FROM NL_SE LIMIT 5000', conn)
    se['klen'] = se['KettoNum'].str.len()
    print('NL_SE KettoNum lengths:', se['klen'].value_counts().to_dict())

    conn.close()

extract_db_stats(db_old, '2006-2015')
extract_db_stats(db_new, '2016-2026')
