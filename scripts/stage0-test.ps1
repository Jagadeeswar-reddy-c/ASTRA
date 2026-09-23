<#
.SYNOPSIS
    ASTRA Stage-0 field test on a Windows PC with one NVIDIA GPU (docs/05-testing/field-test-plan.md).

.DESCRIPTION
    Runs the whole Stage-0 test unattended:
      1. checks the NVIDIA driver and the ASTRA install
      2. downloads llama.cpp (Windows CUDA 12.4 build) and a GGUF model if missing
      3. plans the model with `astra plan` (fit, layer split, speed estimate)
      4. starts llama-server with exactly the command ASTRA generated
      5. measures real generation speed (2 runs)
      6. runs the runtime gate (R01-R05) while the model is generating
      7. writes results.json + report.md and prints PASS / FAIL

    Pass criteria: the model fits, the runtime gate passes, and the measured speed is
    within +/-30 % of ASTRA's estimate.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\stage0-test.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\stage0-test.ps1 -Model 7b -KeepRunning
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\stage0-test.ps1 -ModelPath D:\models\my.gguf -Context 8192
#>
[CmdletBinding()]
param(
    [string]$LabDir = "$env:USERPROFILE\astra-lab",
    [ValidateSet("3b", "7b")][string]$Model = "3b",
    [string]$ModelPath,
    [int]$Context = 4096,
    [int]$Port = 8080,
    [int]$MaxWaitSeconds = 240,
    [switch]$KeepRunning
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest progress bars are very slow on PS 5.1

$Repo = Split-Path -Parent $PSScriptRoot
$Astra = Join-Path $Repo ".venv\Scripts\astra.exe"
$Models = @{
    "3b" = @{ File = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
              Url  = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf" }
    "7b" = @{ File = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
              Url  = "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf" }
}

function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host "`nFAIL: $text" -ForegroundColor Red; exit 1 }
# Windows PowerShell 5.1 turns a native program's stderr into terminating errors when
# redirected; llama.cpp logs to stderr. Run natives with Continue and check exit codes.
function Native([scriptblock]$Block) {
    $saved = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & $Block } finally { $ErrorActionPreference = $saved }
}
function Quote($arg) { if ($arg -match '[\s"]') { '"' + ($arg -replace '"', '\"') + '"' } else { $arg } }

function Download($url, $dest) {
    Write-Host "    downloading $url"
    # curl.exe ships with Windows 10+ and is much faster than Invoke-WebRequest.
    # Download to .part and rename only when complete, so an interrupted download is
    # never mistaken for a finished file on the next run.
    $part = "$dest.part"
    Native { & curl.exe -L --fail --retry 3 -o $part $url }
    if ($LASTEXITCODE -ne 0) { Fail "download failed: $url" }
    Move-Item -Force $part $dest
}

# --------------------------------------------------------------------------- 1. checks
Step "Checking prerequisites"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Fail "nvidia-smi not found: install the NVIDIA driver" }
$gpu = (Native { & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader }) | Select-Object -First 1
Write-Host "    GPU: $gpu"
if (-not (Test-Path $Astra)) {
    Fail "ASTRA not installed. In $Repo run:  py -3.11 -m venv .venv ; .\.venv\Scripts\pip install -e ."
}
$AstraVersion = (Native { & $Astra --version }) | Select-Object -First 1
Write-Host "    $AstraVersion"
New-Item -ItemType Directory -Force -Path $LabDir, "$LabDir\models", "$LabDir\dl" | Out-Null

# ------------------------------------------------------------------- 2. llama.cpp + model
$Server = Join-Path $LabDir "llama\llama-server.exe"
if (-not (Test-Path $Server)) {
    Step "Downloading llama.cpp (Windows, CUDA 12.4)"
    $releases = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
    $release = $releases | Where-Object { $_.assets.name -match '^llama-b\d+-bin-win-cuda-12\.4-x64\.zip$' } | Select-Object -First 1
    if (-not $release) { Fail "no Windows CUDA 12.4 build found in the latest llama.cpp releases" }
    $bin = $release.assets | Where-Object { $_.name -match '^llama-b\d+-bin-win-cuda-12\.4-x64\.zip$' } | Select-Object -First 1
    $rt = $release.assets | Where-Object { $_.name -eq 'cudart-llama-bin-win-cuda-12.4-x64.zip' } | Select-Object -First 1
    Write-Host "    release $($release.tag_name)"
    Download $bin.browser_download_url "$LabDir\dl\llama.zip"
    Expand-Archive -Force "$LabDir\dl\llama.zip" "$LabDir\llama"
    if ($rt) { Download $rt.browser_download_url "$LabDir\dl\cudart.zip"; Expand-Archive -Force "$LabDir\dl\cudart.zip" "$LabDir\llama" }
}
$devices = Native { & $Server --list-devices 2>&1 } | Out-String
if ($devices -notmatch 'CUDA0') { Fail "llama.cpp does not see a CUDA GPU:`n$devices" }
Write-Host "    llama.cpp OK: $((($devices -split "`n") | Where-Object { $_ -match 'CUDA0' }).Trim())"

if ($ModelPath) {
    $Gguf = (Resolve-Path $ModelPath).Path
} else {
    $Gguf = Join-Path "$LabDir\models" $Models[$Model].File
    if (-not (Test-Path $Gguf)) { Step "Downloading model $($Models[$Model].File)"; Download $Models[$Model].Url $Gguf }
}
Write-Host "    model: $Gguf"

# ------------------------------------------------------------------------------ 3. plan
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Out = Join-Path $LabDir "results-$Stamp"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$PlanJson = Join-Path $Out "plan.json"

Step "Planning with ASTRA"
Native { & $Astra plan --gguf $Gguf --context $Context --binary $Server --port $Port --output $PlanJson 2>&1 | ForEach-Object { if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { $_ } } | Write-Host }
$planExit = $LASTEXITCODE
if ($planExit -eq 2) { Fail "astra plan reported an error (see above)" }
$Plan = Get-Content $PlanJson -Raw | ConvertFrom-Json
if (-not $Plan.fits) { Fail "the model does not fit this GPU at context $Context (see the plan above). Try -Context 2048 or a smaller model." }
$Estimate = [double]$Plan.est_decode_tokens_per_s

# ----------------------------------------------------------------------------- 4. engine
Step "Starting llama-server with ASTRA's planned command"
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    Fail "port $Port is already in use (another llama-server?). Stop it or pass -Port."
}
foreach ($kv in $Plan.launch.env.PSObject.Properties) { Set-Item -Path "env:$($kv.Name)" -Value $kv.Value }
$argv = @($Plan.launch.argv)
$argLine = ($argv[1..($argv.Count - 1)] | ForEach-Object { Quote $_ }) -join ' '
$Log = Join-Path $Out "llama-server.log"
$proc = Start-Process -FilePath $argv[0] -ArgumentList $argLine -NoNewWindow -PassThru `
        -RedirectStandardOutput $Log -RedirectStandardError "$Log.err"
$deadline = (Get-Date).AddSeconds($MaxWaitSeconds)
while ($true) {
    if ($proc.HasExited) { Get-Content "$Log.err" -Tail 15; Fail "llama-server exited while loading (log: $Log.err)" }
    try { if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2).status -eq "ok") { break } } catch { }
    if ((Get-Date) -gt $deadline) { $proc | Stop-Process -Force; Fail "llama-server not healthy after $MaxWaitSeconds s" }
    Start-Sleep -Seconds 2
}
Write-Host "    engine healthy on http://127.0.0.1:$Port (pid $($proc.Id))"

try {
    # ---------------------------------------------------------------------- 5. speed
    Step "Measuring generation speed (2 x 300 tokens)"
    $body = @{ prompt = "Explain in detail how a PCIe packet switch lets several GPUs share one link."
               n_predict = 300; cache_prompt = $false } | ConvertTo-Json
    $speeds = @()
    foreach ($run in 1..2) {
        $r = Invoke-RestMethod "http://127.0.0.1:$Port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300
        $speeds += [double]$r.timings.predicted_per_second
        Write-Host ("    run {0}: {1:N1} tok/s" -f $run, $r.timings.predicted_per_second)
    }
    $Measured = ($speeds | Measure-Object -Average).Average
    $Delta = if ($Estimate -gt 0) { ($Measured - $Estimate) / $Estimate * 100 } else { 0 }

    # ----------------------------------------------------------------- 6. runtime gate
    Step "Runtime gate (R01-R05) while the model is generating"
    $load = Start-Job -ScriptBlock {
        param($port)
        $b = @{ prompt = "Write a long technical essay about GPU memory bandwidth."; n_predict = 1200 } | ConvertTo-Json
        Invoke-RestMethod "http://127.0.0.1:$port/completion" -Method Post -Body $b -ContentType "application/json" -TimeoutSec 600 | Out-Null
    } -ArgumentList $Port
    Start-Sleep -Seconds 3
    $GateMd = Join-Path $Out "runtime-gate.md"
    Native { & $Astra validate --phase runtime --plan $PlanJson --samples 8 --format md --output $GateMd 2>&1 | ForEach-Object { if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { $_ } } | Write-Host }
    $GateExit = $LASTEXITCODE
    Stop-Job $load -ErrorAction SilentlyContinue; Remove-Job $load -Force -ErrorAction SilentlyContinue
} finally {
    if (-not $KeepRunning) { $proc | Stop-Process -Force -ErrorAction SilentlyContinue }
}

# -------------------------------------------------------------------------- 7. report
$Pass = ($GateExit -eq 0) -and ([math]::Abs($Delta) -le 30)
$Result = [ordered]@{
    timestamp        = (Get-Date).ToString("s")
    gpu              = $gpu
    astra            = $AstraVersion
    model            = $Plan.model.name
    gguf             = $Gguf
    context          = $Context
    layers           = $Plan.model.n_layers
    planned_required = $Plan.devices[0].required_bytes
    estimate_tok_s   = [math]::Round($Estimate, 1)
    measured_tok_s   = [math]::Round($Measured, 1)
    delta_percent    = [math]::Round($Delta, 1)
    runtime_gate     = if ($GateExit -eq 0) { "PASS" } else { "FAIL" }
    result           = if ($Pass) { "PASS" } else { "FAIL" }
}
$Result | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $Out "results.json")
@"
# ASTRA Stage-0 result: $($Result.result)

| Item | Value |
|---|---|
| GPU | $($Result.gpu) |
| ASTRA | $($Result.astra) |
| Model | $($Result.model) (context $Context) |
| Estimated speed | $($Result.estimate_tok_s) tok/s |
| Measured speed | $($Result.measured_tok_s) tok/s ($($Result.delta_percent) %) |
| Runtime gate | $($Result.runtime_gate) (details: runtime-gate.md) |

Pass criteria: fits, runtime gate PASS, speed within +/-30 % of the estimate.
"@ | Set-Content -Encoding utf8 (Join-Path $Out "report.md")

Step "Result"
$Result.GetEnumerator() | ForEach-Object { "    {0,-17} {1}" -f $_.Key, $_.Value } | Write-Host
Write-Host "`n    Files: $Out" -ForegroundColor Gray
if ($KeepRunning) {
    Write-Host "`n    Engine still running (pid $($proc.Id)). Console:  $Astra ui --plan `"$PlanJson`"  -> http://127.0.0.1:9838/"
}
if ($Pass) { Write-Host "`nPASS" -ForegroundColor Green; exit 0 } else { Write-Host "`nFAIL" -ForegroundColor Red; exit 1 }
