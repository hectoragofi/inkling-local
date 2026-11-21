"""Assemble a mixed GGUF: python -I mix.py OUT.gguf BASE.gguf [SRC.gguf REGEX]...
Every tensor comes from BASE unless its name matches a REGEX, then type+data come from that SRC (later pairs win).
Metadata (KV) is copied bytewise from BASE. Tensor names/dims must match."""
import re, struct, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdump import rd, rstr, rval
def parse(path):
    f = open(path, "rb"); assert f.read(4) == b"GGUF"
    ver = rd(f, "I"); nt, nkv = rd(f, "Q"), rd(f, "Q"); kv0 = f.tell(); align = 32
    for _ in range(nkv):
        k = rstr(f); v = rval(f, rd(f, "I"))
        if k == "general.alignment": align = v
    kv1 = f.tell(); ts = []
    for _ in range(nt):
        name = rstr(f); dims = [rd(f, "Q") for _ in range(rd(f, "I"))]; tt = rd(f, "I"); off = rd(f, "Q")
        ts.append(dict(name=name, dims=dims, type=tt, off=off, path=path))
    base = (f.tell() + align - 1) // align * align; end = os.path.getsize(path) - base
    so = sorted(ts, key=lambda t: t["off"])
    for a, b in zip(so, so[1:] + [dict(off=end)]): a["size"] = b["off"] - a["off"]
    for t in ts: t["abs"] = base + t["off"]
    f.seek(kv0); kv = f.read(kv1 - kv0); f.close()
    return ver, nkv, kv, ts, align
out, basep, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
ver, nkv, kv, ts, align = parse(basep)
for sp, rx in zip(pairs[0::2], pairs[1::2]):
    src = {t["name"]: t for t in parse(sp)[3]}; n = 0
    for i, t in enumerate(ts):
        if re.search(rx, t["name"]):
            s = src[t["name"]]; assert s["dims"] == t["dims"], t["name"]; ts[i] = s; n += 1
    print(f"{n} tensors from {os.path.basename(sp)} ~ {rx}")
pad = lambda n: (-n) % align
off = 0; infos = b""
for t in ts:
    nb = t["name"].encode()
    infos += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(t["dims"])) + b"".join(struct.pack("<Q", d) for d in t["dims"]) + struct.pack("<IQ", t["type"], off)
    t["noff"] = off; off += t["size"] + pad(t["size"])
head = b"GGUF" + struct.pack("<IQQ", ver, len(ts), nkv) + kv + infos
with open(out, "wb") as o:
    o.write(head); o.write(b"\0" * pad(len(head))); d0 = o.tell(); files = {}
    for t in ts:
        assert o.tell() - d0 == t["noff"]
        f = files.setdefault(t["path"], open(t["path"], "rb")); f.seek(t["abs"]); left = t["size"]
        while left:
            b = f.read(min(left, 64 << 20)); assert b; o.write(b); left -= len(b)
        o.write(b"\0" * pad(t["size"]))
print("wrote", out, "%.2f GB" % (os.path.getsize(out) / 1e9))
