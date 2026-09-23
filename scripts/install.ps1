<#
.SYNOPSIS
    Install ASTRA on Windows and (optionally) run it: one step for new users.

.DESCRIPTION
    1. finds Python 3.11+ (py launcher or python on PATH)
    2. creates .venv in the repository and installs ASTRA into it
    3. checks the NVIDIA driver
    4. runs `astra auto` (detect GPUs -> choose model -> download -> launch -> verify -> console)

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -NoRun
#>
[CmdletBinding()]
param(
    [switch]$NoRun,             # install only
    [string[]]$AutoArgs = @()   # extra arguments for `astra auto`, e.g. -AutoArgs '--no-ui'
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
function Step($t) { Write-Host "`n==> $t" -ForegroundColor Cyan }
function Fail($t) { Write-Host "`nERROR: $t" -ForegroundColor Red; exit 1 }
function Native([scriptblock]$Block) {
    $saved = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & $Block } finally { $ErrorActionPreference = $saved }
}

Step "Finding Python 3.11 or newer"
$candidates = @(@("py", "-3.13"), @("py", "-3.12"), @("py", "-3.11"), @("python"), @("python3"))
$python = $null
foreach ($c in $candidates) {
    if (-not (Get-Command $c[0] -ErrorAction SilentlyContinue)) { continue }
    $exe = $c[0]; $pre = @($c | Select-Object -Skip 1)
    $ok = Native { & $exe @pre -c "import sys; print(sys.version_info >= (3, 11))" 2>$null }
    if ("$ok".Trim() -eq "True") { $python = $c; break }
}
if (-not $python) {
    Fail "Python 3.11+ not found. Install it (e.g.  winget install Python.Python.3.12 ), reopen the terminal, and run this again."
}
$pyExe = $python[0]; $pyArgs = @($python | Select-Object -Skip 1)
Write-Host "    using: $($python -join ' ') ($(Native { & $pyExe @pyArgs --version }))"

Step "Creating the virtual environment and installing ASTRA"
$venv = Join-Path $Repo ".venv"
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    Native { & $pyExe @pyArgs -m venv $venv }
    if ($LASTEXITCODE -ne 0) { Fail "could not create $venv" }
}
Native { & "$venv\Scripts\python.exe" -m pip install --quiet --upgrade pip }
Native { & "$venv\Scripts\python.exe" -m pip install --quiet -e $Repo }
if ($LASTEXITCODE -ne 0) { Fail "pip install failed" }
$astra = "$venv\Scripts\astra.exe"
Write-Host "    $(Native { & $astra --version })  ->  $astra"

Step "Checking the NVIDIA driver"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Native { & nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader } |
        ForEach-Object { Write-Host "    GPU $_" }
} else {
    Write-Host "    nvidia-smi not found: install the NVIDIA driver before running ASTRA." -ForegroundColor Yellow
    $NoRun = $true
}

Write-Host "`nInstalled. Use ASTRA from any terminal with:" -ForegroundColor Green
Write-Host "    $astra auto          # detect GPUs, choose + download a model, launch, verify, open console"
Write-Host "    $astra --help        # all commands"
if ($NoRun) { exit 0 }

Step "Running: astra auto $($AutoArgs -join ' ')"
& $astra auto @AutoArgs
exit $LASTEXITCODE
