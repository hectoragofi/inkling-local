# decode-only breakdown: longest stretch of the trace between prompt-processing events (big dequant / sgemm bursts)
import csv, collections, sys
_src = open('prof.py').read()
exec(_src.split("rows = [r for r in rows if r[0] >= t_end")[0])
exec("def fam" + _src.split("def fam")[1].split("agg = collections")[0])
cuts = [s for s, d, n in rows if 'dequantize_block' in n and d > 1e7]
segs = []; prev = rows[0][0]
for c in cuts + [rows[-1][0] + 1]:
    segs.append((prev, c)); prev = c
a, b = max(segs, key=lambda x: x[1] - x[0])
seg = [r for r in rows if a + 2e7 < r[0] < b - 1e6]
ntok = sum(1 for r in seg if 'flash_attn_ext_vec' in r[2]) / 48
span = (seg[-1][0] + seg[-1][1] - seg[0][0]) / 1e6
agg = collections.defaultdict(float)
for s, d, n in seg: agg[fam(n)] += d
tot = sum(agg.values()) / 1e6
print(f"decode stretch {span:.0f} ms, {ntok:.1f} tokens, {span/ntok:.2f} ms/token; kernels {tot/ntok:.2f} ms/token; GPU idle {(span-tot)/ntok:.2f} ms/token")
for k, v in sorted(agg.items(), key=lambda x: -x[1])[:16]:
    print(f"  {k:36s} {v/1e6/ntok:6.2f} ms/token")
