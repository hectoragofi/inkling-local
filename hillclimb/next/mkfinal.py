# python -I mkfinal.py <name> <mkcfg args...>: like mkcfg.py, then --load-mode mlock and --cache-ram 2048 (the final flag lines)
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1]
subprocess.run([sys.executable, "-I", os.path.join(HERE, "mkcfg.py")] + sys.argv[1:], check=True, stdout=subprocess.DEVNULL)
p = os.path.join(HERE, "configs", name + ".json")
c = json.load(open(p))
a = c["args"]; a[a.index("--load-mode") + 1] = "mlock"; a += ["--cache-ram", "2048"]
json.dump(c, open(p, "w"), indent=1)
print(name, " ".join(a[2:]), c["env"])
