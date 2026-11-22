# queue 1: 120K decode A/B with controls, then smoke
cd /c/Users/hecto/models/inkling/hillclimb/fresh
for c in base-rb fresh-gqa0 fresh-gqa1 base-rb fresh-gqa1; do python -I bench120.py run configs/$c.json 6 2>&1 | grep -E "MEDIAN|text:|exit|busy|Error" ; done
python -I bench120.py smoke configs/fresh-gqa1.json 2>&1 | grep -E "SMOKE|Error"
