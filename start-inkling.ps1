# Start Inkling 2 Mini (tuned pick from the 21 Nov hill-climb) with the chat UI at http://127.0.0.1:8080
# Fallback if the 4060 drives a display (less than ~7.9 GB VRAM free): run with -Safe
param([switch]$Safe)
$m = "$PSScriptRoot\custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer.gguf"
$ub = "80"; $ot = 'blk\.([0-5]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU'
if ($Safe) { $ub = "64"; $ot = 'blk\.([0-7]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU' }
& "$PSScriptRoot\llama\llama-server.exe" -m $m --host 127.0.0.1 --port 8080 --alias inkling -np 1 `
  -ngl 99 -c 32768 --jinja -ctk q8_0 -ctv q8_0 --load-mode none -ub $ub -ot $ot `
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0
