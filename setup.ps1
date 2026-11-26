# One-shot setup for Inkling 2 Mini on an 8 GB NVIDIA GPU (Windows).
#   .\setup.ps1             build mainline llama.cpp + patches\next if the toolchain is there (~33 tok/s), else use the prebuilt
#   .\setup.ps1 -Prebuilt   skip the build, use unsloth's prebuilt llama.cpp (~26 tok/s, no ngram speculation)
#   .\setup.ps1 -Clean      afterwards delete the two source GGUFs (~29 GB) that the tuned file was mixed from
# Safe to re-run: every step skips work that is already done. Downloads resume.
# Needs: NVIDIA GPU + driver, Python 3.10+, ~60 GB free (~31 GB after -Clean), git.
# Fork build also needs: CUDA Toolkit 13.x, Visual Studio 2022 (C++ workload), CMake.
param([switch]$Prebuilt, [switch]$Clean)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$HF       = "https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main"
$MAINLINE = "$HF/Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file"
$PREBUILT_URL = "https://github.com/unslothai/llama.cpp/releases/download/b12588-mix-c0c3de8/app-b12588-mix-c0c3de8-windows-x64-cuda13-newer.zip"
$CUDART   = "https://github.com/ggml-org/llama.cpp/releases/download/b12606/cudart-llama-bin-win-cuda-13.5-x64.zip"
$UPSTREAM = "https://github.com/ggml-org/llama.cpp"
$BASE_REV = "c212d25ad34ff1fdf775793cda07e009560cf66b"   # master 23 Nov 2026; patches\next\*.patch go on top

$IQ3  = "$root\Inkling-2-Mini-UD-IQ3_XXS.gguf"
$Q4   = "$root\Inkling-2-Mini-UD-Q4_K_XL.gguf"
$PICK = "$root\custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer.gguf"
$PICKM = "$root\custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer-mainline.gguf"

function Step($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }
function Have($c) { [bool](Get-Command $c -ErrorAction SilentlyContinue) }
function Get-File($url, $out) {
  if (Test-Path $out) { Write-Host "have $(Split-Path $out -Leaf)"; return }
  Write-Host "downloading $(Split-Path $out -Leaf)"
  & curl.exe -fL --retry 5 -C - -o "$out.part" $url
  if ($LASTEXITCODE -ne 0) { Fail "download failed: $url (re-run to resume)" }
  Move-Item "$out.part" $out -Force
}
function Copy-Cudart($dir) {
  $z = "$root\cudart\cudart-llama-bin-win-cuda-13.5-x64.zip"
  New-Item -ItemType Directory -Force "$root\cudart" | Out-Null
  Get-File $CUDART $z
  Expand-Archive $z -DestinationPath $dir -Force
}

# ---------------------------------------------------------------- checks
Step "Checking the machine"
if (-not (Have nvidia-smi)) { Fail "nvidia-smi not found: install the NVIDIA driver" }
$gpu = (& nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | Select-Object -First 1).Split(",") | ForEach-Object { $_.Trim() }
Write-Host "GPU: $($gpu[0]), $($gpu[1]), compute $($gpu[2])"
$cc = $gpu[2] -replace "\.", ""
if ([int]$cc -lt 86) { Write-Host "warning: tuned and tested on sm_89 (RTX 40). Older GPUs may not run the NVFP4 kernels." -ForegroundColor Yellow }
if (-not (Have curl.exe)) { Fail "curl.exe not found (ships with Windows 10 1803+)" }
$py = $null
foreach ($c in @("py", "python", "python3")) { if (Have $c) { $py = $c; break } }
if (-not $py) { Fail "Python 3.10+ not found" }
$free = (Get-PSDrive ($root.Substring(0, 1))).Free / 1GB
$need = 0
if (-not (Test-Path $PICK) -and -not (Test-Path $PICKM)) { $need += 43 }
if (-not $Prebuilt -and -not (Test-Path $PICKM)) { $need += 13 }
Write-Host ("free disk: {0:N0} GB, need about {1} GB more" -f $free, $need)
if ($free -lt $need) { Fail "not enough disk space" }

# Fork build is possible when the toolchain is installed.
$cudaPath = $env:CUDA_PATH
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsVer = $null
if (Test-Path $vswhere) { $vsVer = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property catalog_productLineVersion }
$canBuild = $cudaPath -and (Test-Path "$cudaPath\bin\nvcc.exe") -and $vsVer -and (Have cmake) -and (Have git)
$haveBuilt = (Test-Path "$root\build-ib\bin\Release\llama-server.exe") -or (Test-Path "$root\build-rb\bin\Release\llama-server.exe")
$useFork = (-not $Prebuilt) -and ($canBuild -or $haveBuilt)
if (-not $Prebuilt -and -not $useFork) {
  Write-Host "CUDA Toolkit / Visual Studio 2022 C++ / CMake / git not all found: using the prebuilt (~26 tok/s instead of ~33)." -ForegroundColor Yellow
}

# ---------------------------------------------------------------- python
Step "Python environment"
if (-not (Test-Path "$root\.venv\Scripts\python.exe")) { & $py -m venv "$root\.venv" }
$vpy = "$root\.venv\Scripts\python.exe"
& $vpy -c "import numpy" 2>$null
if ($LASTEXITCODE -ne 0) { & $vpy -m pip install --quiet numpy; if ($LASTEXITCODE -ne 0) { Fail "pip install numpy failed" } }
Write-Host "ok"

# ---------------------------------------------------------------- model
Step "Tuned model file (UD-IQ3_XXS + native NVFP4 experts of layers 0-5, 18-19)"
New-Item -ItemType Directory -Force "$root\custom" | Out-Null
if (Test-Path $PICK) { Write-Host "have $(Split-Path $PICK -Leaf)" }
elseif ($useFork -and (Test-Path $PICKM)) { Write-Host "have $(Split-Path $PICKM -Leaf)" }
else {
  Get-File "$HF/Inkling-2-Mini-UD-IQ3_XXS.gguf" $IQ3
  Get-File "$HF/Inkling-2-Mini-UD-Q4_K_XL.gguf" $Q4
  & $vpy -I "$root\hillclimb\mix.py" "$PICK.part" $IQ3 $Q4 '^blk\.([0-5]|1[89])\.ffn_.*_exps\.'
  if ($LASTEXITCODE -ne 0) { Fail "mix.py failed" }
  Move-Item "$PICK.part" $PICK -Force
}

# ---------------------------------------------------------------- runtime
if ($useFork) {
  Step "llama.cpp: mainline $($BASE_REV.Substring(0, 9)) + patches\next (Inkling speed patches)"
  $bdir = "$root\build-ib"
  if (Test-Path "$bdir\bin\Release\llama-server.exe") { Write-Host "have build-ib\" }
  elseif ((-not $canBuild) -and (Test-Path "$root\build-rb\bin\Release\llama-server.exe")) { $bdir = "$root\build-rb"; Write-Host "have build-rb\ (older fork build; install the toolchain and re-run for build-ib)" }
  else {
    $src = "$root\src\llama.cpp-ib"
    if (-not (Test-Path "$src\.git")) {
      & git clone -q --filter=blob:none --no-checkout -c core.longpaths=true $UPSTREAM $src; if ($LASTEXITCODE -ne 0) { Fail "git clone failed" }
    }
    if (Test-Path "$src\.git\rebase-apply") { & git -C $src am --abort }
    & git -C $src -c core.longpaths=true checkout -q -f --detach $BASE_REV; if ($LASTEXITCODE -ne 0) { Fail "git checkout $BASE_REV failed" }
    $patches = Get-ChildItem "$root\patches\next\00*.patch" | Sort-Object Name | ForEach-Object { $_.FullName }
    & git -C $src -c core.longpaths=true -c user.name=inkling-setup -c user.email=setup@localhost am -q --whitespace=nowarn @patches
    if ($LASTEXITCODE -ne 0) { Fail "applying patches\next failed" }
    $gen = "Visual Studio 17 2022"
    if ($vsVer -ne "2022") { $gen = "Visual Studio $((& $vswhere -latest -property installationVersion).Split('.')[0]) $vsVer" }
    & cmake -S $src -B $bdir -G $gen -A x64 -T "cuda=$cudaPath" "-DCUDAToolkit_ROOT=$cudaPath" `
      -DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=$cc" -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
    if ($LASTEXITCODE -ne 0) { Fail "cmake configure failed (re-run with -Prebuilt to skip the build)" }
    & cmake --build $bdir --config Release -j $env:NUMBER_OF_PROCESSORS --target llama-server llama-bench
    if ($LASTEXITCODE -ne 0) { Fail "build failed (re-run with -Prebuilt to skip the build)" }
  }
  if (-not (Test-Path "$bdir\bin\Release\cudart64_13.dll")) { Copy-Cudart "$bdir\bin\Release" }

  Step "Converting the tuned file to the mainline inkling2_loop layout"
  if (Test-Path $PICKM) { Write-Host "have $(Split-Path $PICKM -Leaf)" }
  else {
    $head = "$root\cudart\mainline-head.bin"
    if (-not (Test-Path $head)) {
      & curl.exe -fsL -r 0-16777215 -o $head $MAINLINE; if ($LASTEXITCODE -ne 0) { Fail "header download failed" }
    }
    & $vpy -I "$root\hillclimb\to_mainline.py" $PICK "$PICKM.part" $head $MAINLINE
    if ($LASTEXITCODE -ne 0) { Fail "to_mainline.py failed" }
    Move-Item "$PICKM.part" $PICKM -Force
  }
}
else {
  Step "llama.cpp: unsloth prebuilt b12588-mix-c0c3de8 (cuda13-newer)"
  if (Test-Path "$root\llama\llama-server.exe") { Write-Host "have llama\" }
  else {
    Get-File $PREBUILT_URL "$root\llama.zip"
    Expand-Archive "$root\llama.zip" -DestinationPath "$root\llama" -Force
  }
  # The bundle ships without the CUDA runtime; without it CUDA silently doesn't load.
  if (-not (Test-Path "$root\llama\cudart64_13.dll")) { Copy-Cudart "$root\llama" }
}

# ---------------------------------------------------------------- cleanup
if ($Clean) {
  Step "Cleaning up"
  foreach ($f in @($IQ3, $Q4)) { if (Test-Path $f) { Remove-Item $f; Write-Host "deleted $(Split-Path $f -Leaf)" } }
  if ($useFork -and (Test-Path $PICKM) -and (Test-Path $PICK)) { Remove-Item $PICK; Write-Host "deleted $(Split-Path $PICK -Leaf) (the fork uses the mainline copy)" }
}

Step "Done"
Write-Host "Start it:  .\start-inkling.ps1        (add -Safe if this GPU also drives your display)"
Write-Host "Then open  http://127.0.0.1:8080       (OpenAI-compatible API at /v1, model name 'inkling')"
