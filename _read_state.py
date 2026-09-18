# -*- coding: utf-8 -*-
"""读取 TraeCode state.vscdb 查找认证信息"""
import sqlite3
import os
import shutil
import tempfile

src = r'C:\Users\16270\AppData\Roaming\TRAE SOLO CN\User\globalStorage\state.vscdb'
tmp = os.path.join(tempfile.gettempdir(), 'trae_state.db')
shutil.copy2(src, tmp)
conn = sqlite3.connect(tmp)
c = conn.cursor()

# List all tables
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = c.fetchall()
print('Tables:', [t[0] for t in tables])

# Check ItemTable for auth keys
c.execute("SELECT key FROM ItemTable WHERE key LIKE '%auth%' OR key LIKE '%token%' OR key LIKE '%icube%' OR key LIKE '%credential%' OR key LIKE '%secret%' OR key LIKE '%login%'")
rows = c.fetchall()
print(f'\nAuth-related keys: {len(rows)}')
for r in rows:
    print(f'  {r[0]}')

# Also check for any key with 'trae' or 'session'
c.execute("SELECT key FROM ItemTable WHERE key LIKE '%trae%' OR key LIKE '%session%' OR key LIKE '%checkin%'")
rows2 = c.fetchall()
print(f'\nTrae/session keys: {len(rows2)}')
for r in rows2:
    print(f'  {r[0]}')

conn.close()
os.remove(tmp)
