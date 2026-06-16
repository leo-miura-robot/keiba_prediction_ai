import sqlite3

db_old = r'D:\keiba\new_jra_2006-2015_fixed_20260616_015045\keiba.db'

conn = sqlite3.connect(db_old)
for tbl in ['NL_SE', 'NL_O2']:
    cols = conn.execute(f"PRAGMA table_info({tbl})").fetchall()
    print(f'{tbl} columns:', [c[1] for c in cols])
conn.close()
