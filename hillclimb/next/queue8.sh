cd /c/Users/hecto/models/inkling/hillclimb/next
python -I ab.py configs/rbf-128k.json configs/fin-128k.json short pp deep check > logs/ab-fin-128k.txt 2>&1
python -I ab.py configs/rbf-32k.json configs/fin-32k.json short pp check > logs/ab-fin-32k.txt 2>&1
bash sweep.sh "fin-32k-dec short pp check" "fin-32k-spec short check" "fin-64k short pp check" "fin-64k-c32 short pp check" > logs/sweep8.txt 2>&1
python -I kld.py build-next-snap/ibg2 ibg2-std > logs/kld8.txt 2>&1
python -I kld.py build-next-snap/ibg2 ibg2-ub1-q4 -ub 1 -ctk q4_0 -ctv q4_0 >> logs/kld8.txt 2>&1
python -I kld.py build-next-snap/ibg2 ibg2-ub8-q8 -ub 8 -ctk q8_0 -ctv q8_0 >> logs/kld8.txt 2>&1
python -I kld.py build-next-snap/ibg2 ibg2-ub8-q8-mc256 -ub 8 -ctk q8_0 -ctv q8_0 --moe-cache-mib 256 --moe-cache-cpu-share 0.95 >> logs/kld8.txt 2>&1
python -I kld.py build-next-snap/ibg2 ibg2-loop1 --override-kv inkling2_loop.loop.count=int:1 --override-kv inkling2_loop.loop.varying=bool:true >> logs/kld8.txt 2>&1
echo queue8-done
