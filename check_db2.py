import sqlite3
import pandas as pd
import json

db_old = r'D:\keiba\new_jra_2006-2015_fixed_20260616_015045\keiba.db'
db_new = r'D:\keiba\new_jra_2016-2026_fixed\keiba.db'

def audit_jrdb(path, name):
    conn = sqlite3.connect(path)
    print(f'\n=== JRDB: {name} ===')
    
    # Check SE (Results)
    try:
        se = pd.read_sql('SELECT LENGTH(RaceCD) as rlen, LENGTH(KettoNum) as klen, FukuPay1, FukuPay2, FukuPay3, KakuteiJyuni FROM NL_SE LIMIT 1000', conn)
        print('NL_SE RaceCD len:', se['rlen'].value_counts().to_dict())
        print('NL_SE KettoNum len:', se['klen'].value_counts().to_dict())
        
        # Parse Payouts
        pays = []
        for c in ['FukuPay1', 'FukuPay2', 'FukuPay3']:
            pays.extend(pd.to_numeric(se[c], errors='coerce').dropna().tolist())
        pays = [p for p in pays if p > 0]
        if pays:
            print(f'NL_SE Payout Unit Example (raw numeric): min={min(pays)}, max={max(pays)}')
        else:
            print('NL_SE Payouts: None parsed as positive numbers')
    except Exception as e:
        print('NL_SE error:', e)

    # Check O2 (Place Odds)
    try:
        o2 = pd.read_sql('SELECT RaceCD, UmaBan1, FukuOddsMin, FukuOddsMax FROM NL_O2 LIMIT 1000', conn)
        odds = pd.to_numeric(o2['FukuOddsMin'], errors='coerce').dropna()
        if len(odds) > 0:
            print(f'NL_O2 FukuOddsMin: min={odds.min()}, max={odds.max()} (Raw string sample: {o2["FukuOddsMin"].dropna().head().tolist()})')
    except Exception as e:
        print('NL_O2 error:', e)
        
    # Check RA (Race)
    try:
        ra = pd.read_sql('SELECT SyussoTosu FROM NL_RA LIMIT 1000', conn)
        tosu = pd.to_numeric(ra['SyussoTosu'], errors='coerce').dropna()
        if len(tosu) > 0:
            print(f'NL_RA SyussoTosu: min={tosu.min()}, max={tosu.max()}')
    except Exception as e:
        print('NL_RA error:', e)

    conn.close()

audit_jrdb(db_old, '2006-2015')
audit_jrdb(db_new, '2016-2026')
