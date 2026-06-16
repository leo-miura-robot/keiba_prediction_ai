import sqlite3
import pandas as pd

db_old = r'D:\keiba\new_jra_2006-2015_fixed_20260616_015045\keiba.db'
db_new = r'D:\keiba\new_jra_2016-2026_fixed\keiba.db'

def get_tables(path):
    conn = sqlite3.connect(path)
    res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    conn.close()
    return [r[0] for r in res]

print("Tables in 2006 DB:", get_tables(db_old))

def audit_db(path, name):
    conn = sqlite3.connect(path)
    print(f'\n=== DB: {name} ===')
    
    tables = get_tables(path)
    if 'race_info' in tables:
        try:
            races = pd.read_sql('SELECT payout_place, entry_count FROM race_info LIMIT 1000', conn)
            print('Sample payout_place:', races['payout_place'].dropna().head().tolist())
        except Exception as e:
            print('race_info error:', e)
    elif 'races' in tables:
         try:
            races = pd.read_sql('SELECT * FROM races LIMIT 1', conn)
            print('Races cols:', races.columns.tolist())
         except Exception as e:
            print('races error:', e)

    if 'horse_race_info' in tables:
        pass
    elif 'entries' in tables:
         try:
            entries = pd.read_sql('SELECT * FROM entries LIMIT 1', conn)
            print('Entries cols:', entries.columns.tolist())
         except Exception as e:
            print('entries error:', e)

    conn.close()

audit_db(db_old, '2006-2015')
audit_db(db_new, '2016-2026')
