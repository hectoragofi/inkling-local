# python -I mkcfg.py <name> <build dir> <mode 32k|128k> [ub] [cpu-expert-layer regex] [extra args...] [ENV=VAL ...]
# writes configs/<name>.json for bench.py
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\hecto\models\inkling"
name, build, mode = sys.argv[1:4]
rest = sys.argv[4:]
env = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in rest if "=" in a and not a.startswith("-")}
rest = [a for a in rest if not ("=" in a and not a.startswith("-"))]
ub = rest[0] if rest else ("80" if mode == "32k" else "64")
cpu = rest[1] if len(rest) > 1 else ("([0-5]|1[89]|2[0-3])" if mode == "32k" else "([0-7]|1[89]|2[0-3])")
extra = rest[2:]
ctx, kv = {"32k": ("32768", "q8_0"), "64k": ("65536", "q8_0"), "128k": ("131072", "q4_0")}[mode]
args = ["-m", ROOT + r"\custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer-mainline.gguf", "-ngl", "99", "-c", ctx, "--jinja",
        "-ctk", kv, "-ctv", kv, "--load-mode", "none", "-ub", ub, "-ot", r"blk\." + cpu + r"\.ffn_.*_exps\.=CPU"] + extra
cfg = {"name": name, "exe": os.path.join(ROOT, build, r"bin\Release\llama-server.exe"), "args": args, "env": env}
json.dump(cfg, open(os.path.join(HERE, "configs", name + ".json"), "w"), indent=1)
print(name, " ".join(args[2:]), env)
