# Start Inkling 2 Mini (tuned pick). Chat UI + API at http://127.0.0.1:8080
#   -Safe      the 4060 also drives a display (less than ~7.9 GB VRAM free)
#   -Long      128K context (q4_0 KV, experts of 0-6 + 18-23 on CPU; 0-7 on older builds)
#   -Compress  also start the LLMLingua-2 compressing proxy: point apps at http://127.0.0.1:8081/v1
#   -Greedy    temperature 0 by default (clients can still override per request). Copy-heavy code then hits 55-90 tok/s.
#   -Effort    thinking effort when the client does not set one: minimal, low, medium, high (model default), xhigh.
#              Lower = fewer thinking tokens per reply (faster, less context used), at some cost in hard-task quality.
#   -NoKeeper  serve llama-server directly on 8080, without the thinking keeper (see below)
# Thinking keeper (keeper.py): Inkling expects its own earlier thinking in the history, and most apps do not send it
# back. The keeper sits on 8080, remembers each reply's thinking and puts it back; llama-server moves to 8079.
#   -NoSpec    no ngram-mod speculation (greedy output is then identical to plain decoding)
# ngram-mod speculation: copy-heavy replies (code edits) about 2x faster, free text unchanged. Needs a self-built server.
# Server, best first: build-ib\ (mainline llama.cpp + patches\next: ~33 tok/s, 419 tok/s prefill at 32K),
# build-rb\ (realugbun inkling2-speed fork, ~31 tok/s), else unsloth's prebuilt (~26 tok/s).
# Weights are pinned in RAM (--load-mode mlock): under RAM pressure Windows paged the CPU experts out and decode fell to <1 tok/s. Needs ~10 GB free RAM at start.
param([switch]$Safe, [switch]$Long, [switch]$Compress, [switch]$Greedy, [switch]$NoKeeper, [switch]$NoSpec,
      [ValidateSet("", "minimal", "low", "medium", "high", "xhigh")][string]$Effort = "")
$ib = "$PSScriptRoot\build-ib\bin\Release\llama-server.exe"
$fork = "$PSScriptRoot\build-rb\bin\Release\llama-server.exe"
$mainline = "$PSScriptRoot\custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer-mainline.gguf"
$spec = @(); $kind = "prebuilt"
if (Test-Path $ib) { $exe = $ib; $m = $mainline; $kind = "ib" }
elseif (Test-Path $fork) { $exe = $fork; $m = $mainline; $kind = "rb" }
else { $exe = "$PSScriptRoot\llama\llama-server.exe"; $m = "$PSScriptRoot\custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer.gguf" }
if ($kind -ne "prebuilt" -and -not $NoSpec) { $spec = @("--spec-type", "ngram-mod") }
if (-not (Test-Path $exe) -or -not (Test-Path $m)) { Write-Host "Not set up yet: run .\setup.ps1 first." -ForegroundColor Red; exit 1 }

# RAM check. The model needs ~10 GB of free RAM and ~17 GB of commit. A browser can hold many GB of commit as
# shared GPU memory (seen: 16 GB), which no process list shows; then the load fails or Windows pages the weights out.
try {
  $c = (Get-Counter '\Memory\Available MBytes', '\Memory\Committed Bytes', '\Memory\Commit Limit' -ErrorAction Stop).CounterSamples
  $free = [math]::Round($c[0].CookedValue / 1024, 1); $room = [math]::Round(($c[2].CookedValue - $c[1].CookedValue) / 1GB, 1)
  if ($free -lt 10 -or $room -lt 18) {
    Write-Host "Low memory: $free GB RAM free, $room GB of commit left (want 10 and 18). Biggest users of shared GPU memory:" -ForegroundColor Yellow
    (Get-Counter '\GPU Process Memory(*)\Total Committed' -ErrorAction Stop).CounterSamples | Sort-Object CookedValue -Descending |
      Select-Object -First 3 | ForEach-Object {
        $id = [int]([regex]::Match($_.Path, 'pid_(\d+)').Groups[1].Value)
        Write-Host ("  {0} (pid {1}): {2:N1} GB" -f (Get-Process -Id $id -ErrorAction SilentlyContinue).Name, $id, ($_.CookedValue / 1GB)) -ForegroundColor Yellow
      }
  }
} catch {}

$temp = "1.0"; if ($Greedy) { $temp = "0" }
$ctx = "32768"; $kv = "q8_0"; $ub = "80"; $ot = 'blk\.([0-5]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU'
$extra = @()
if ($kind -eq "ib") {
  # Measured in hillclimb\next (A/B/A, KLD unchanged): 32K 32.7 tok/s + 419 prefill; 128K 29.1, 25.1 at 107K deep.
  $ub = "384"; $extra += @("--moe-cache-mib", "256", "--moe-cache-cpu-share", "0.95")
  if ($Safe) { $ub = "80"; $extra = @(); $ot = 'blk\.([0-7]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU' }
  if ($Long) { $ctx = "131072"; $kv = "q4_0"; $ub = "1024"; $extra = @(); $ot = 'blk\.([0-6]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU' }
} else {
  if ($Safe) { $ub = "64"; $ot = 'blk\.([0-7]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU' }
  if ($Long) { $ctx = "131072"; $kv = "q4_0"; $ub = "64"; $ot = 'blk\.([0-7]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU' }
}
if ($Effort) { $extra += @("--reasoning-effort", $Effort) }

$port = "8080"; $keeper = $null
if (-not $NoKeeper) {
  $py = "$PSScriptRoot\.venv\Scripts\python.exe"
  if (-not (Test-Path $py)) { $py = (Get-Command python -ErrorAction SilentlyContinue).Source }
  if ($py) {
    $port = "8079"
    $keeper = Start-Process $py -ArgumentList @("`"$PSScriptRoot\keeper.py`"", "--port", "8080", "--upstream", "http://127.0.0.1:8079") -WindowStyle Hidden -PassThru
    Write-Host "thinking keeper on http://127.0.0.1:8080 (llama-server itself on 8079)"
  } else { Write-Host "Python not found: starting without the thinking keeper." -ForegroundColor Yellow }
}
if ($Compress) {
  $pa = @("proxy.py")
  Start-Process "$PSScriptRoot\compress\.venv\Scripts\python.exe" -ArgumentList $pa -WorkingDirectory "$PSScriptRoot\compress" -WindowStyle Minimized
  Write-Host "proxy starting on http://127.0.0.1:8081 (compressing)"
}
# --cache-ram 2048: the server keeps old conversations' KV in RAM (default up to 8 GB). On 32 GB that pushes the
# pinned weights toward the pagefile; 2 GB still holds about three full 32K conversations.
try {
  & $exe -m $m --host 127.0.0.1 --port $port --alias inkling -np 1 `
    -ngl 99 -c $ctx --jinja -ctk $kv -ctv $kv --load-mode mlock -ub $ub -ot $ot --cache-ram 2048 `
    @spec @extra --temp $temp --top-p 1.0 --top-k 0 --min-p 0.0
} finally {
  if ($keeper -and -not $keeper.HasExited) { Stop-Process -Id $keeper.Id -ErrorAction SilentlyContinue }
}
