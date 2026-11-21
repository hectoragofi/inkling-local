"""Decode/prefill ceilings for Inkling 2 Mini GGUF with a GPU/CPU expert split.
Reads tensor sizes from the GGUF header. Looped core layers [core_start, core_end) run num_loops times.

  python ceil.py MODEL.gguf [--cpu-layers 0-5,18-23] [--gpu-bw 220] [--cpu-bw 52.5] [--tflops 28] [--ctx 32768]
"""
import re, struct, sys
from collections import defaultdict

T = {0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),9:(32,36),10:(256,84),11:(256,110),
     12:(256,144),13:(256,176),14:(256,210),15:(256,292),16:(256,66),17:(256,74),18:(256,98),19:(256,50),
     20:(32,18),21:(256,110),22:(256,82),23:(256,136),24:(1,1),25:(1,2),26:(1,4),27:(1,8),28:(1,8),29:(256,56),
     30:(1,2),34:(256,54),35:(256,66),39:(32,17),40:(16,9)}

def rd(f, fmt): return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
def rstr(f): return f.read(rd(f, "Q")).decode("utf-8", "replace")
def rval(f, t):
    simple = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
    if t in simple: return rd(f, simple[t])
    if t == 8: return rstr(f)
    at, n = rd(f, "I"), rd(f, "Q"); return [rval(f, at) for _ in range(n)]

def load(path):
    f = open(path, "rb"); assert f.read(4) == b"GGUF"
    rd(f, "I"); nt, nkv = rd(f, "Q"), rd(f, "Q"); meta = {}
    for _ in range(nkv):
        k = rstr(f); meta[k] = rval(f, rd(f, "I"))
    layers = defaultdict(lambda: defaultdict(int)); other = defaultdict(int)
    for _ in range(nt):
        name = rstr(f); dims = [rd(f, "Q") for _ in range(rd(f, "I"))]; tt = rd(f, "I"); rd(f, "Q")
        n = 1
        for d in dims: n *= d
        be, bb = T.get(tt, (1, 0)); b = n // be * bb
        m = re.match(r"blk\.(\d+)\.(.*)", name)
        if m: layers[int(m.group(1))]["exp" if "_exps" in m.group(2) else "dense"] += b
        else: other[name] += b
    return meta, layers, other

def parse_layers(s):
    out = set()
    for part in filter(None, s.split(",")):
        a, _, b = part.partition("-"); out |= set(range(int(a), int(b or a) + 1))
    return out

def main():
    a = sys.argv[1:]; opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    meta, layers, other = load(a[0])
    cpu = parse_layers(opt("--cpu-layers", ""))
    gbw, cbw, tf, ctx = float(opt("--gpu-bw", 220)), float(opt("--cpu-bw", 52.5)), float(opt("--tflops", 28)), int(opt("--ctx", 32768))
    cs, ce, loops = meta["inkling2.loop.core_start"], meta["inkling2.loop.core_end"], meta["inkling2.num_loops"]
    k, E = meta["inkling2.expert_used_count"], meta["inkling2.expert_count"]
    rep = lambda L: loops if cs <= L < ce else 1
    gpu_tok = cpu_tok = 0.0; gpu_res = cpu_res = 0.0
    for L, d in layers.items():
        gpu_tok += d["dense"] * rep(L); gpu_res += d["dense"]
        e = d["exp"] * k / E * rep(L)
        if L in cpu: cpu_tok += e; cpu_res += d["exp"]
        else: gpu_tok += e; gpu_res += d["exp"]
    gpu_tok += other.get("output.weight", 0); gpu_res += other.get("output.weight", 0)
    cpu_res += other.get("token_embd.weight", 0)  # embedding lookup stays on CPU in llama.cpp
    # KV: global-attention layer applications only (SWA layers hold 1024-token window)
    pat = meta["inkling2.attention.sliding_window_pattern"]; nkv = meta["inkling2.attention.head_count_kv"][0]
    hd = meta["inkling2.attention.key_length"]; win = meta["inkling2.attention.sliding_window"]
    g, s = pat.count(False), pat.count(True)
    kv_f16 = (g * ctx + s * min(ctx, win)) * 2 * nkv * hd * 2
    kv_tok = g * 2 * nkv * hd * 2 * ctx / 2  # avg read at half-full ctx, f16
    t = gpu_tok / gbw / 1e9 + cpu_tok / cbw / 1e9 + kv_tok / gbw / 1e9
    tot = sum(sum(d.values()) for d in layers.values()) + sum(other.values())
    print(f"{a[0].split(chr(92))[-1]}: {tot/1e9:.2f} GB, cpu expert layers {sorted(cpu) or 'none'}")
    print(f"  resident GPU {gpu_res/1e9:.2f} GB + KV f16@{ctx//1024}K {kv_f16/1e9:.2f} GB (q8_0 {kv_f16/2e9:.2f}) | CPU {cpu_res/1e9:.2f} GB")
    print(f"  bytes/token: GPU {gpu_tok/1e9:.3f} GB, CPU {cpu_tok/1e9:.3f} GB -> c1 decode ceiling {1/t:.0f} tok/s"
          f" (all-GPU would be {1/((gpu_tok+cpu_tok+kv_tok)/gbw/1e9):.0f})")
    act = (gpu_tok + cpu_tok)  # proxy unused; FLOPs from active params: 5.4B effective
    print(f"  prefill compute ceiling ~{0.5*tf*1e12/(2*5.4e9):.0f} tok/s (+ PCIe weight streaming of {cpu_res/1e9:.1f} GB CPU experts per ubatch at 13.3 GB/s)")

main()
