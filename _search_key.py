# -*- coding: utf-8 -*-
"""搜索 TraeCode main.js 中的加密密钥服务名"""
import re

with open(r'D:\software\TRAE SOLO CN\resources\app\out\main.js', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Search for service name patterns related to secret storage
patterns = [
    r'secretStorageService',
    r'keytar',
    r'getPassword',
    r'setPassword',
    r'dpapi',
    r'applicationName.*secret',
]

for pat in patterns:
    idx = content.find(pat)
    if idx >= 0:
        snippet = content[max(0, idx-80):idx+120]
        print(f'\n=== {pat} @ {idx} ===')
        print(snippet)
    else:
        print(f'{pat}: not found')
