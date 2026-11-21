# Start Inkling 2 Mini (tuned pick from the 21 Nov hill-climb). Chat UI + API at http://127.0.0.1:8080
#   -Safe      the 4060 also drives a display (less than ~7.9 GB VRAM free)
#   -Long      128K context (q4_0 KV, experts of 0-7 + 18-23 on CPU): ~20.6 tok/s instead of ~26
#   -Compress  also start the LLMLingua-2 compressing proxy: point apps at http://127.0.0.1:8081/v1
param([switch]$Safe, [switch]$Long, [switch]$Compress)
$m = "$PSScriptRoot\custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer.gguf"
$ctx = "32768"; $kv = "q8_0"; $ub = "80"; $ot = 'blk\.([0-5]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU'
if ($Safe) { $ub = "64"; $ot = 'blk\.([0-7]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU' }
if ($Long) { $ctx = "131072"; $kv = "q4_0"; $ub = "64"; $ot = 'blk\.([0-7]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU' }
if ($Compress) {
  Start-Process "$PSScriptRoot\compress\.venv\Scripts\python.exe" -ArgumentList "proxy.py" -WorkingDirectory "$PSScriptRoot\compress" -WindowStyle Minimized
  Write-Host "compressing proxy starting on http://127.0.0.1:8081 (takes ~1 min to load the first time)"
}
& "$PSScriptRoot\llama\llama-server.exe" -m $m --host 127.0.0.1 --port 8080 --alias inkling -np 1 `
  -ngl 99 -c $ctx --jinja -ctk $kv -ctv $kv --load-mode none -ub $ub -ot $ot `
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0
