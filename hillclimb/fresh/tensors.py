import sys, collections, re
sys.path.insert(0, r"C:\Users\hecto\models\inkling\src\llama.cpp-rb\gguf-py")
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
for k,v in r.fields.items():
    if k.startswith(("general.arch",)) or ".block_count" in k or "expert" in k or "head" in k or "context" in k or "embedding" in k or "feed_forward" in k or "loop" in k.lower() or "window" in k or "rope" in k:
        try: val = v.contents()
        except Exception: val='?'
        if isinstance(val,list) and len(val)>30: val=str(val[:30])+'...'
        print(k, val)
agg = collections.defaultdict(lambda:[0,0,set()])
per_layer = collections.defaultdict(int)
for t in r.tensors:
    n = t.name; b = int(t.n_bytes)
    m = re.match(r"blk\.(\d+)\.(.*)", n)
    key = m.group(2) if m else n
    agg[key][0]+=b; agg[key][1]+=1; agg[key][2].add(t.tensor_type.name)
    if m: per_layer[int(m.group(1))]+=b
    if m and m.group(1) in ("0","6"): print(n, list(t.shape), t.tensor_type.name, b)
    if not m: print(n, list(t.shape), t.tensor_type.name, b)
print("---agg (GB)")
for k,(b,c,ty) in sorted(agg.items(), key=lambda x:-x[1][0]): print(f"{k:40s} {b/1e9:8.3f} n={c} {sorted(ty)}")
print("total", sum(v[0] for v in agg.values())/1e9)
