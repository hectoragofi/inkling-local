# Router skew from llama-imatrix output files (per-expert token counts of each blk.N.ffn_gate_up_exps).
#   python route_stats.py a.imatrix [b.imatrix ...]
# Per layer: share of routed tokens taken by the top 25% / 40% / 50% of experts (ranked on the pooled counts), and
# how well a hot set ranked on one file covers each other file (does a static hot set generalise).
import re, sys
import numpy as np
sys.path.insert(0, r"C:\Users\hecto\models\inkling\src\llama.cpp-next\gguf-py")
from gguf import GGUFReader

def load(path):
    r = GGUFReader(path)
    out = {}
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.ffn_gate_up_exps\.weight\.counts", t.name)
        if m:
            out[int(m.group(1))] = np.array(t.data, dtype=np.float64).reshape(-1)
    return out

files = sys.argv[1:]
data = [load(f) for f in files]
layers = sorted(data[0])
E = len(data[0][layers[0]])
fr = [0.10, 0.25, 0.40, 0.50]
print(f"{len(files)} files, {E} experts per layer")
print("layer  tokens   " + "  ".join(f"top{int(f*100)}%" for f in fr) + "   cross-file coverage of top40% (ranked on file i, measured on file j)")
for l in layers:
    pooled = sum(d[l] for d in data)
    order = np.argsort(-pooled)
    tot = pooled.sum()
    shares = [pooled[order[:int(E*f)]].sum() / tot for f in fr]
    cross = []
    for i, di in enumerate(data):
        hot = np.argsort(-di[l])[:int(E*0.40)]
        for j, dj in enumerate(data):
            if i != j:
                cross.append(dj[l][hot].sum() / dj[l].sum())
    cr = f"min {min(cross):.3f} mean {np.mean(cross):.3f}" if cross else ""
    print(f"{l:5d}  {int(tot):7d}   " + "  ".join(f"{s:6.3f}" for s in shares) + "   " + cr)
