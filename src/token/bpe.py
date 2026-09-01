from collections import defaultdict

# 语料：词 -> 出现次数（模拟真实语料里的词频）
corpus = {
    "hug": 10,
    "pug": 5,
    "pun": 12,
    "bun": 4,
    "hugs": 5,
}

# 每个词先用"字符列表"表示（字符就是最初的词元）
words = {w: list(w) for w, f in corpus.items()}
print(words)
freqs = dict(corpus)  # 词频


def get_pair_stats(words, freqs):
    """统计当前切分下，所有相邻词元对的频率（按词频加权）"""
    stats = defaultdict(int)
    for w, toks in words.items():
        for i in range(len(toks) - 1):
            stats[(toks[i], toks[i + 1])] += freqs[w]
    return dict(stats)


def apply_merge(toks, pair):
    """把 toks 中所有相邻的 pair 合并成一个词元"""
    out, i = [], 0
    while i < len(toks):
        if i + 1 < len(toks) and (toks[i], toks[i + 1]) == pair:
            out.append(pair[0] + pair[1])
            i += 2
        else:
            out.append(toks[i])
            i += 1
    return out


merges = []  # 按顺序记录学到的合并规则，推理时按同样顺序回放

print("初始词表:", sorted({c for w in words for c in words[w]}))
print()

n_merges = 5  # 每合并一次词表 +1，等价于"词表达到目标大小就停"
for step in range(1, n_merges + 1):
    stats = get_pair_stats(words, freqs)
    print(stats)
    best = max(stats.items(), key=lambda kv: (kv[1], kv[0]))  # 频率最高的一对
    pair, count = best
    merged = pair[0] + pair[1]
    merges.append(pair)

    print(f"第 {step} 次合并: {pair} -> '{merged}'  (频率 {count})")
    for w, toks in sorted(words.items()):
        words[w] = apply_merge(toks, pair)
        print(f"    {w:>5} -> {words[w]}")
    print()

# 最终词表 = 初始字符 + 所有合并产物
final_vocab = {c for w in corpus for c in w} | {a + b for a, b in merges}
print("最终词表 (大小 %d):" % len(final_vocab), sorted(final_vocab))
print()


def tokenize(word):
    """推理：按学到的合并顺序切分一个新词"""
    toks = list(word)
    for pair in merges:
        toks = apply_merge(toks, pair)
    return toks


for w in ["hug", "pugs", "punish", "bunny"]:
    print(f"新词 {w!r:8} -> {tokenize(w)}")
