# python bench runs in sequence: sweep.sh "<config> <what...>" ...   prints the RESULT lines (trimmed)
cd /c/Users/hecto/models/inkling/hillclimb/next
for job in "$@"; do
  set -- $job
  c=$1; shift
  python -I bench.py configs/$c.json "$@" 2>&1 | grep -a "RESULT\|exited\|Error" | python -I -c "
import sys, json
for l in sys.stdin:
    if l.startswith('RESULT '):
        r = json.loads(l[7:]); print({k: v for k, v in r.items() if not k.endswith('_text') and k not in ('ts',)}, flush=True)
    else: print(l.strip()[:300], flush=True)"
done
