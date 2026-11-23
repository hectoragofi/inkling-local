# KLD gate: python -I kld.py <build dir> <log name> [extra llama-perplexity args...] [ENV=VAL ...]
# Same as the earlier gates: 20 chunks x 512 of quality\corpus.txt vs quality\base-q4kxl.kld (UD-Q4_K_XL logits),
# -b 2048 (4 sequences per batch), 32K-mode expert split. Add "-ub 1" to run the decode (VEC) attention path.
import os, re, subprocess, sys
ROOT = r"C:\Users\hecto\models\inkling"
HERE = os.path.dirname(os.path.abspath(__file__))
build, name = sys.argv[1], sys.argv[2]
rest = sys.argv[3:]
env = dict(os.environ)
for a in [a for a in rest if "=" in a and not a.startswith("-")]:
    k, v = a.split("=", 1); env[k] = v
extra = [a for a in rest if not ("=" in a and not a.startswith("-"))]
if os.path.exists(os.path.join(ROOT, ".gpu-busy")):
    sys.exit("GPU busy flag present")
exe = os.path.join(ROOT, build, r"bin\Release\llama-perplexity.exe")
args = [exe, "-m", ROOT + r"\custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer-mainline.gguf", "-f", ROOT + r"\quality\corpus.txt",
        "--kl-divergence-base", ROOT + r"\quality\base-q4kxl.kld", "--kl-divergence", "-c", "512", "--chunks", "20", "-b", "2048",
        "-ngl", "99", "--load-mode", "none", "-ot", r"blk\.([0-5]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU"] + extra
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
log = os.path.join(HERE, "logs", name + ".kld.log")
with open(log, "w") as f:
    rc = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
s = open(log, errors="replace").read()
kld = re.findall(r"Mean\s+KLD:\s+([\d.]+)\s+±\s+([\d.]+)", s)
top = re.findall(r"Same top p:\s+([\d.]+)\s+±\s+([\d.]+)", s)
ppl = re.findall(r"Mean PPL\(Q\)\s+:\s+([\d.]+)", s)
print("KLD", name, "rc", rc, "kld", kld[-1] if kld else None, "same_top", top[-1] if top else None, "ppl", ppl[-1] if ppl else None)
