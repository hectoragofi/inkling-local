# Analyse an nsys cuda_gpu_trace CSV of a llama-bench decode window.
# usage: python prof.py trace.csv [n_last_seconds]
# Picks the last window of the trace (decode tokens), groups kernels by family, reports per-token ms,
# GPU busy fraction and the idle gaps (CPU expert work / syncs).
import csv, sys, re, collections
rows = []
with open(sys.argv[1], newline='', encoding='utf-8', errors='replace') as f:
    for r in csv.DictReader(f):
        try:
            s = int(r['Start (ns)']); d = int(r['Duration (ns)'])
        except (KeyError, ValueError):
            continue
        rows.append((s, d, r.get('Name', '')))
rows.sort()
win = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
t_end = rows[-1][0] + rows[-1][1]
rows = [r for r in rows if r[0] >= t_end - win * 1e9]
t0 = rows[0][0]
span = (t_end - t0) / 1e6
# token boundaries: the LM head is the largest mul_mat_vec over Q5_K... use the softcap/argmax? count by a marker kernel
def fam(n):
    n2 = n.split('(')[0].split('<')[0].strip()
    if 'memcpy' in n.lower() or 'Memcpy' in n: return 'memcpy'
    if 'flash_attn_ext_vec' in n: return 'fa_vec'
    if 'flash_attn_combine' in n: return 'fa_combine'
    if 'flash_attn' in n: return 'fa_other'
    if 'inkling2_relbias' in n: return 'relbias_mask'
    if 'inkling2' in n: return 'inkling2_' + n2.replace('inkling2_', '').split('_f32')[0]
    if 'mul_mat_vec_q' in n:
        m = re.search(r'ggml_type\)(\d+)', n)
        return 'mmvq' + ('_moe' if 'true' in n.split(',')[2:3] else '') + (f'_t{m.group(1)}' if m else '')
    if 'mul_mat_vec_f' in n: return 'mmvf'
    if 'mul_mat_q' in n: return 'mmq'
    if 'quantize' in n: return 'quantize'
    return n2[:40]
agg = collections.defaultdict(lambda: [0, 0])
for s, d, n in rows:
    a = agg[fam(n)]; a[0] += d; a[1] += 1
busy = 0; last = t0; gaps = []
for s, d, n in rows:
    if s > last:
        gaps.append(s - last)
    busy += max(0, s + d - max(s, last)) if s + d > last else 0
    last = max(last, s + d)
ntok = float(sys.argv[3]) if len(sys.argv) > 3 else None
print(f"window {span:.1f} ms, kernels {len(rows)}, busy {busy/1e6:.1f} ms ({100*busy/1e6/span:.0f}%)")
big = [g for g in gaps if g > 50_000]
print(f"gaps>50us: {len(big)} totalling {sum(big)/1e6:.1f} ms; gaps<=50us total {sum(g for g in gaps if g<=50_000)/1e6:.1f} ms")
div = ntok or 1
for k, (d, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:30]:
    print(f"{k:40s} {d/1e6/div:8.3f} ms{'/tok' if ntok else ''}  n={c/div:7.1f}  avg {d/c/1e3:7.1f} us")
