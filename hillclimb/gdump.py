"""Dump GGUF tensor table: python -I gdump.py MODEL.gguf [regex]  -> per-tensor name, type, dims, bytes (from offsets)"""
import re, struct, sys, os
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
    ts = []
    for _ in range(nt):
        name = rstr(f); dims = [rd(f, "Q") for _ in range(rd(f, "I"))]; tt = rd(f, "I"); off = rd(f, "Q")
        ts.append([name, tt, dims, off])
    align = meta.get("general.alignment", 32)
    base = (f.tell() + align - 1) // align * align
    end = os.path.getsize(path) - base
    so = sorted(ts, key=lambda t: t[3])
    for a, b in zip(so, so[1:] + [[None, 0, [], end]]): a.append(b[3] - a[3])
    return meta, ts, base
if __name__ == "__main__":
    meta, ts, base = load(sys.argv[1]); pat = sys.argv[2] if len(sys.argv) > 2 else "."
    if pat == "META":
        for k, v in meta.items():
            s = str(v); print(k, "=", s[:120])
    else:
        for n, tt, d, off, sz in ts:
            if re.search(pat, n): print(f"{n:45s} t={tt:3d} {str(d):28s} {sz/1e6:9.1f} MB")
