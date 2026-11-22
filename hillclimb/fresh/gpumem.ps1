$p = Get-Process llama-server,llama-bench,llama-perplexity -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $p) { "no llama process"; exit }
$s = (Get-Counter "\GPU Process Memory(pid_$($p.Id)*)\Shared Usage","\GPU Process Memory(pid_$($p.Id)*)\Dedicated Usage").CounterSamples
$sh = ($s | ? Path -like '*shared usage' | Measure-Object CookedValue -Sum).Sum/1MB
$de = ($s | ? Path -like '*dedicated usage' | Measure-Object CookedValue -Sum).Sum/1MB
"pid $($p.Id) dedicated {0:N0} MiB shared {1:N0} MiB" -f $de,$sh
