import json

stats = json.load(open('oscar_all_stats.json', 'r'))
stats_sorted = dict(sorted(stats.items(), key=lambda t: t[1], reverse=True))
char_count = sum(stats.values())
print("Total number of characters: {:,}".format(char_count))
stats_sorted = {k: v / char_count for k, v in stats_sorted.items()}
min_prefix = min([k for k in range(len(stats_sorted)) if sum(list(stats_sorted.values())[:k]) >= 0.9993])
print("Minimal top k found: k = {}".format(min_prefix))

top_k_char_stats = list(stats_sorted.items())[:min_prefix]
top_k_char_stats_sorted = sorted(top_k_char_stats)
print("<unk> probability after taking top k: {:.4f}%".format(100 - 100 * sum([v for _, v in top_k_char_stats])))

print("Writing characters to dictionary ...")
with open('dict.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(
        ["unused{} 1".format(str(i).zfill(2)) for i in range(100)] + [ch + " 1" for ch, _ in top_k_char_stats_sorted]))
print("Done!")
