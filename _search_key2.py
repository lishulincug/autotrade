# -*- coding: utf-8 -*-
"""搜索 TraeCode 加密密钥服务名"""
with open(r'D:\software\TRAE SOLO CN\resources\app\out\main.js', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Find more context around getPassword
idx = content.find('getPassword')
if idx >= 0:
    snippet = content[max(0, idx-200):idx+400]
    print('=== getPassword context ===')
    print(snippet)

# Search for the service name used with keytar
import re
# Look for patterns like "vscode-" or "trae-" followed by a version
matches = list(re.finditer(r'(?:service|Service)\s*[=:]\s*["\'`][^"\'`]{1,60}["\'`]', content))
for m in matches[:15]:
    print(f'\nService match @ {m.start()}: {m.group()[:120]}')

# Also search for where applicationName is used
matches2 = list(re.finditer(r'applicationName', content))
for m in matches2[:5]:
    snippet = content[max(0, m.start()-50):m.start()+100]
    print(f'\nappName @ {m.start()}: ...{snippet}...')
