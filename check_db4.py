import sqlite3
db_old = r'D:\keiba\new_jra_2006-2015_fixed_20260616_015045\keiba.db'
conn = sqlite3.connect(db_old)
cols = conn.execute("PRAGMA table_info(NL_HR)").fetchall()
print('NL_HR columns:', [c[1] for c in cols])
conn.close()
