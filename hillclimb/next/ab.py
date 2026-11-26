# Interleaved gate: python -I ab.py <control.json> <candidate.json> [what ...]   (control, candidate, control)
# Prints each run and a verdict per metric: a win beats both controls; a regression is >3% below the worse control.
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ctrl, cand = sys.argv[1], sys.argv[2]
what = sys.argv[3:] or ["short", "pp"]
runs = []
for cfg in (ctrl, cand, ctrl):
    out = subprocess.run([sys.executable, "-I", os.path.join(HERE, "bench.py"), cfg] + what, capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        print(out.stdout[-2000:], out.stderr[-2000:]); sys.exit(f"bench failed for {cfg}")
    rec = json.loads(line[-1][7:]); runs.append(rec)
    print({k: v for k, v in rec.items() if not k.endswith("_text") and k != "ts"}, flush=True)
c1, b, c2 = runs
for m in ("short", "pp", "deep", "deep_suffix_ms", "vram_mib", "compute_mib"):
    if m not in b: continue
    lo, hi = min(c1[m], c2[m]), max(c1[m], c2[m])
    better_low = m in ("deep_suffix_ms", "vram_mib", "compute_mib")
    if better_low:
        verdict = "WIN" if b[m] < lo else ("REGRESS" if b[m] > hi*1.03 else "same")
    else:
        verdict = "WIN" if b[m] > hi else ("REGRESS" if b[m] < lo*0.97 else "same")
    print(f"{m:15s} ctrl {c1[m]} / {c2[m]}  cand {b[m]}  {verdict}  ({(b[m]/((c1[m]+c2[m])/2)-1)*100:+.1f}% vs mean ctrl)")
for r in runs:
    if "check" in r: print(r["name"], "check", r["check"])
