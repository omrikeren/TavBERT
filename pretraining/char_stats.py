import json
from collections import defaultdict

chars = defaultdict(int)

with open('oscar_all.txt', 'r', encoding='utf-8') as f:
    for ind, line in enumerate(f):
        if ind % 1e05 == 0:
            open('oscar_all_stats.json', 'w', encoding='utf-8').write(json.dumps(chars))
        for ch in line:
            chars[ch] += 1

open('oscar_all_stats.json', 'w', encoding='utf-8').write(json.dumps(chars))
