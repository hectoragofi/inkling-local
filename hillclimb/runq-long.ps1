cd C:\Users\hecto\models\inkling\hillclimb
$bench='C:\Users\hecto\AppData\Local\Temp\claude\99e816625e03ff51\c19e5664-fd2f-4073-a4d1-74867e7bbaa4\scratchpad\ios\skills\inference-optimisation\scripts\bench.py'
foreach ($n in $args) { python -I $bench run configs\$n.json --long 30000 *>> logs\q3.txt }
