# Fresh ceilings: bytes read per decode token by device, from the GGUF tensor sizes.
import sys, re, collections
sys.path.insert(0, r"C:\Users\hecto\models\inkling\src\llama.cpp-rb\gguf-py")
from gguf import GGUFReader
r = GGUFReader(r"C:\Users\hecto\models\inkling\custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer-mainline.gguf")
GPU_BW, CPU_BW = 220e9, 52.5e9
CORE = range(6, 18); PASSES = 3
def evals(l): return PASSES if l in CORE else 1
lay = collections.defaultdict(lambda: collections.defaultdict(int)); glob = {}
for t in r.tensors:
    m = re.match(r"blk\.(\d+)\.(.*)", t.name)
    if m: lay[int(m.group(1))][m.group(2)] += int(t.n_bytes)
    else: glob[t.name] = int(t.n_bytes)
def run(cpu_layers, ctx, kv_bytes_tok_layer=1152, label=""):
    gpu_dense = glob["output.weight"]; gpu_exp = cpu_exp = 0; exp_vram = 0; dense_vram = sum(glob.values()) - glob["token_embd.weight"]
    for l, d in lay.items():
        e = d["ffn_gate_up_exps.weight"] + d["ffn_down_exps.weight"]
        dense = sum(v for k, v in d.items() if "exps.weight" not in k)
        gpu_dense += dense * evals(l); dense_vram += dense
        act = e * 8 / 128 * evals(l)
        if l in cpu_layers: cpu_exp += act
        else: gpu_exp += act; exp_vram += e
    n_full_evals = sum(evals(l) for l in (5, 11, 17, 23))
    kv = n_full_evals * ctx * kv_bytes_tok_layer
    mask = n_full_evals * ctx * 24 * 2 * 2  # f16 per-head mask written + read
    swa = (48 - n_full_evals) * 1024 * kv_bytes_tok_layer * 2  # q8? approx
    g = gpu_dense + gpu_exp + kv + mask + swa
    tg, tc = g / GPU_BW * 1e3, cpu_exp / CPU_BW * 1e3
    print(f"{label:28s} ctx={ctx:6d} GPU: dense {gpu_dense/1e9:.3f} exp {gpu_exp/1e9:.3f} kv {kv/1e9:.3f} mask {mask/1e9:.3f} -> {tg:5.2f} ms | CPU exp {cpu_exp/1e9:.3f} -> {tc:5.2f} ms | serial {tg+tc:5.2f} ms = {1e3/(tg+tc):5.1f} tok/s | overlap-ideal {1e3/max(tg,tc):5.1f} | VRAM w {(dense_vram+exp_vram)/1e9:.2f} GB")
    return tg, tc
base = set(range(0, 8)) | set(range(18, 24))
for ctx in (0, 32768, 107000, 120000):
    run(base, ctx, label="baseline128K placement")
run(set(range(0, 6)) | set(range(18, 24)), 0, kv_bytes_tok_layer=2176, label="32K placement (q8 kv)")
run(set(), 0, label="all-GPU (not fitting)")
run(set(range(24)), 0, label="all experts CPU")
