# Launch / stop llama-server detached for bench.py.
#   srv.ps1 start <name> <args...>   -> starts server, logs to logs\<name>.log
#   srv.ps1 stop
#   srv.ps1 mem                      -> prints bytes: llama-server private RAM + GPU VRAM used
# no param() block: PowerShell would bind llama flags like -c to it by prefix
$cmd = $args[0]; $name = $args[1]; $rest = @($args | Select-Object -Skip 2)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$exe = if ($env:INK_EXE) { $env:INK_EXE } else { "$HOME\models\inkling\llama\llama-server.exe" }
# INK_SET="A=1;B=2" sets env vars for the server (WMI children do not inherit ours); INK_EXE overrides the binary
$envp = if ($env:INK_SET) { (($env:INK_SET -split ";") | ForEach-Object { "set `"$_`"&& " }) -join "" } else { "" }
switch ($cmd) {
  "start" {
    New-Item -ItemType Directory -Force "$root\logs" | Out-Null
    $args2 = @("--port", "8080", "--host", "127.0.0.1", "--alias", "inkling", "--no-webui", "-np", "1") + $rest
    # WMI create: child inherits no handles, so the caller's pipes close immediately
    $line = "cmd /c `"$envp`"$exe`" " + (($args2 | ForEach-Object { if ($_ -match '[|<>&^ ()]') { "`"$_`"" } else { $_ } }) -join " ") + " > `"$root\logs\$name.log`" 2>&1`""
    $r = Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{ CommandLine = $line; CurrentDirectory = $root }
    if ($r.ReturnValue -ne 0) { throw "launch failed: $($r.ReturnValue)" }
  }
  "stop" {
    $p = Get-Process llama-server -ErrorAction SilentlyContinue
    if ($p) { $p | Stop-Process; $p | Wait-Process -Timeout 60 -ErrorAction SilentlyContinue }
  }
  "mem" {
    $p = Get-Process llama-server -ErrorAction SilentlyContinue
    $ram = if ($p) { ($p | Measure-Object PrivateMemorySize64 -Sum).Sum } else { 0 }
    $vram = [int64](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) * 1MB
    [int64]($ram + $vram)
  }
}

