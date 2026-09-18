# -*- coding: utf-8 -*-
"""读取 TraeWork Chromium Cookies"""
import sqlite3
import os
import shutil
import tempfile

# Copy to temp to avoid lock
src = r'C:\Users\16270\AppData\Roaming\TRAE SOLO CN\Network\Cookies'
tmp = os.path.join(tempfile.gettempdir(), 'trae_cookies.db')
shutil.copy2(src, tmp)

conn = sqlite3.connect(tmp)
c = conn.cursor()
# Show all cookies for trae.cn domain
c.execute("SELECT host_key, name, path, expires_utc FROM cookies WHERE host_key LIKE '%trae%' ORDER BY host_key")
rows = c.fetchall()
print(f'Trae cookies: {len(rows)}')
for r in rows:
    print(f'  {r[0]} | {r[1]} | path={r[2]} | expires={r[3]}')

# Also check trae-webview partition
conn.close()
os.remove(tmp)

# Check trae-webview partition
src2 = r'C:\Users\16270\AppData\Roaming\TRAE SOLO CN\Partitions\trae-webview\Network\Cookies'
if os.path.exists(src2):
    tmp2 = os.path.join(tempfile.gettempdir(), 'trae_wv_cookies.db')
    shutil.copy2(src2, tmp2)
    conn2 = sqlite3.connect(tmp2)
    c2 = conn2.cursor()
    c2.execute("SELECT host_key, name, path, expires_utc FROM cookies WHERE host_key LIKE '%trae%' OR host_key LIKE '%icube%' ORDER BY host_key")
    rows2 = c2.fetchall()
    print(f'\nTraeWebview cookies: {len(rows2)}')
    for r in rows2:
        print(f'  {r[0]} | {r[1]} | path={r[2]} | expires={r[3]}')
    conn2.close()
    os.remove(tmp2)
