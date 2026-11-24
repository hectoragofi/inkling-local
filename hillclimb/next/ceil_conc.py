# Decode ceilings from the GGUF tensor sizes and real routing counts (llama-imatrix .counts):
#   python ceil_conc.py <routing.imatrix> [vram_expert_budget_MiB] [kv_tokens]
# Bytes per token at 100% of measured bandwidth (GPU 220 GB/s, CPU 52.5 GB/s, THEORY.md). Core layers 6-17 run 3 passes.
# serial: today's placement (core experts on GPU, outer on CPU), time = GPU + CPU
# concurrent: the hot (GPU) and cold (CPU) experts of each layer evaluation run at the same time, time per layer
#   evaluation = max(GPU dense + hot experts, cold experts); the hot set is chosen greedily over all (layer, expert)
#   pairs within the VRAM budget, by wall-time saved per byte.
import re, sys
import numpy as np
sys.path.insert(0, r"C:\Users\hecto\models\inkling\src\llama.cpp-next\gguf-py")
from gguf import GGUFReader
GPU_BW, CPU_BW = 220e9, 52.5e9
MODEL = r"C:\Users\hecto\models\inkling\custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer-mainline.gguf"
CORE = range(6, 18); PASSES = 3; N_USED = 8; KV_BYTES_PER_TOKEN = 2176  # q8_0 K+V per full-attention layer
rd = GGUFReader(MODEL)
dense = np.zeros(24); expert = np.zeros(24); head = 0.0; full_attn = []
for t in rd.tensors:
    m = re.match(r"blk\.(\d+)\.(.*)", t.name)
    if not m:
        if t.name.startswith("output"): head += int(t.n_bytes)
        continue
    il, rest = int(m.group(1)), m.group(2)
    if "_exps" in rest: expert[il] += int(t.n_bytes) / 128.0  # bytes of one expert (gate_up + down)
    else: dense[il] += int(t.n_bytes)
counts = {}
for t in GGUFReader(sys.argv[1]).tensors:
    m = re.match(r"blk\.(\d+)\.ffn_gate_up_exps\.weight\.counts", t.name)
    if m: counts[int(m.group(1))] = np.array(t.data, dtype=np.float64).reshape(-1)
budget = float(sys.argv[2]) * 2**20 if len(sys.argv) > 2 else 5.2 * 2**30
kv_tok = int(sys.argv[3]) if len(sys.argv) > 3 else 0
visits = np.array([PASSES if il in CORE else 1 for il in range(24)])
p = {il: counts[il] / counts[il].sum() * N_USED for il in range(24)}  # expected uses of each expert per layer evaluation
# KV: 8 full-attention layer evaluations per token (THEORY.md), spread evenly over the evaluations for the max() model
kv_per_eval = 8 * kv_tok * KV_BYTES_PER_TOKEN / visits.sum()
gpu_dense = (dense + kv_per_eval * 0) / GPU_BW
def layer_times(hot):
    # per layer evaluation: GPU = dense + hot experts (+ KV share), CPU = cold experts
    g = dense / GPU_BW + kv_per_eval / GPU_BW + np.array([(p[l] * hot[l]).sum() * expert[l] for l in range(24)]) / GPU_BW
    c = np.array([(p[l] * (1 - hot[l])).sum() * expert[l] for l in range(24)]) / CPU_BW
    return g, c
head_t = head / GPU_BW
# serial, today's placement
hot = {l: np.ones(128) if l in CORE else np.zeros(128) for l in range(24)}
g, c = layer_times(hot)
serial = ((g + c) * visits).sum() + head_t
print(f"serial, today's placement: {serial*1e3:.1f} ms/token = {1/serial:.1f} tok/s  (GPU {((g*visits).sum()+head_t)*1e3:.1f} ms, CPU {(c*visits).sum()*1e3:.1f} ms)")
conc_today = (np.maximum(g, c) * visits).sum() + head_t
print(f"concurrent, today's placement: {conc_today*1e3:.1f} ms = {1/conc_today:.1f} tok/s")
# greedy hot set within the budget
hot = {l: np.zeros(128) for l in range(24)}
used = 0.0
order = sorted(((p[l][e] * visits[l] * expert[l] * (1 / CPU_BW - 1 / GPU_BW) / expert[l], l, e) for l in range(24) for e in range(128)), reverse=True)
def total(hot):
    g, c = layer_times(hot)
    return (np.maximum(g, c) * visits).sum() + head_t
# marginal greedy: take experts in order of cold time saved per byte, but skip a layer once its GPU side is the longer one
g, c = layer_times(hot)
for _, l, e in order:
    if used + expert[l] > budget: continue
    if g[l] >= c[l]: continue
    hot[l][e] = 1; used += expert[l]
    g, c = layer_times(hot)
conc = total(hot)
print(f"concurrent, solved hot set ({used/2**20:.0f} MiB of experts on GPU, budget {budget/2**20:.0f}): {conc*1e3:.1f} ms = {1/conc:.1f} tok/s")
print("per layer: hot experts, GPU ms, CPU ms (one evaluation)")
for l in range(24):
    print(f"  {l:2d} x{visits[l]} hot {int(hot[l].sum()):3d}  gpu {g[l]*1e3:.3f}  cpu {c[l]*1e3:.3f}")
