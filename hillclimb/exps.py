import sys, re, collections
sys.path.insert(0, r"C:\Users\hecto\models\inkling\hillclimb")
import gdump
for p in sys.argv[1:]:
    meta, ts, base = gdump.load(p); print(p)
    L = collections.defaultdict(dict); oth = collections.Counter(); dense = collections.Counter()
    for n, tt, d, off, sz in ts:
        m = re.match(r"blk\.(\d+)\.ffn_(gate_up|down)_exps\.weight", n)
        if m: L[int(m.group(1))][m.group(2)] = (tt, sz)
        elif n.startswith("blk."): dense[int(n.split(".")[1])] += sz
        else: oth[n] += sz
    for i in sorted(L): print(f"  {i:2d} gu t{L[i]['gate_up'][0]:<2d} {L[i]['gate_up'][1]/1e6:6.0f}  dn t{L[i]['down'][0]:<2d} {L[i]['down'][1]/1e6:6.0f}  dense {dense[i]/1e6:5.1f}")
    print("  other:", {k: round(v/1e6) for k, v in oth.items()}, "dense total", round(sum(dense.values())/1e6))
