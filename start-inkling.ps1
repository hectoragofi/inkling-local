# Start Inkling 2 Mini (tuned pick). Chat UI + API at http://127.0.0.1:8080
#   -Safe      the 4060 also drives a display (less than ~7.9 GB VRAM free)
#   -Long      128K context (q4_0 KV, experts of 0-7 + 18-23 on CPU)
#   -Compress  also start the LLMLingua-2 compressing proxy: point apps at http://127.0.0.1:8081/v1
#   -Offload   also start the proxy with Modal prefill for long prompts (>= 20K new tokens): point apps at :8081/v1
#              needs cloud\.venv + `modal deploy cloud/modal_prefill.py` once (see README)
# Uses the realugbun inkling2-speed fork (build-rb\) with the mainline-layout pick when built, else the unsloth prebuilt.
param([switch]$Safe, [switch]$Long, [switch]$Compress, [switch]$Offload)
$fork = "$PSScriptRoot\build-rb\bin\Release\llama-server.exe"
if (Test-Path $fork) { $exe = $fork; $m = "$PSScriptRoot\custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer-mainline.gguf" }
else { $exe = "$PSScriptRoot\llama\llama-server.exe"; $m = "$PSScriptRoot\custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer.gguf" }
$ctx = "32768"; $kv = "q8_0"; $ub = "80"; $ot = 'blk\.([0-5]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU'
if ($Safe) { $ub = "64"; $ot = 'blk\.([0-7]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU' }
if ($Long) { $ctx = "131072"; $kv = "q4_0"; $ub = "64"; $ot = 'blk\.([0-7]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU' }
if ($Compress -or $Offload) {
  $pa = @("proxy.py")
  if (-not $Compress) { $pa += "--no-compress" }
  if ($Offload) { $pa += @("--offload-min-tokens", "20000", "--offload-ctx", $ctx, "--offload-kv", $kv) }
  Start-Process "$PSScriptRoot\compress\.venv\Scripts\python.exe" -ArgumentList $pa -WorkingDirectory "$PSScriptRoot\compress" -WindowStyle Minimized
  Write-Host "proxy starting on http://127.0.0.1:8081 (compress=$Compress offload=$Offload)"
}
New-Item -ItemType Directory -Force "$PSScriptRoot\slots" | Out-Null
& $exe -m $m --host 127.0.0.1 --port 8080 --alias inkling -np 1 `
  -ngl 99 -c $ctx --jinja -ctk $kv -ctv $kv --load-mode none -ub $ub -ot $ot --slot-save-path "$PSScriptRoot\slots" `
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0
