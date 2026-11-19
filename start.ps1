<#
  Starts Unmix. Ctrl+C stops everything.

    .\start.ps1                     # backend on this computer + frontend (http://localhost:5173)
    .\start.ps1 -Fake               # instant fake "models" (UI work without the weights)
    .\start.ps1 -Remote <url>       # frontend only, using a backend running elsewhere
                                    # (the Colab notebook in colab/ prints the URL and key)

  The backend's torch / onnxruntime build is chosen from your hardware
  (backend/scripts/pick_accel.py): CUDA for NVIDIA GPUs, DirectML on other
  Windows GPUs, CPU otherwise. Override with -Accel cpu|dml|cuda|cuda12.
#>
param(
    [switch]$Fake,
    [string]$Remote = "",
    [string]$Key = "",
    [ValidateSet("", "cpu", "dml", "cuda", "cuda12")][string]$Accel = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$tools = if ($Remote) { @("npm") } else { @("uv", "npm", "ffmpeg") }
foreach ($tool in $tools) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool was not found on PATH. See README.md > Prerequisites." }
}
if (-not (Test-Path (Join-Path $root "frontend\node_modules"))) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
    Push-Location (Join-Path $root "frontend"); npm install; Pop-Location
}

function Start-Frontend {
    Write-Host "Starting frontend on http://localhost:5173 ..." -ForegroundColor Cyan
    Push-Location (Join-Path $root "frontend")
    try { npm run dev } finally { Pop-Location }
}

# ------------------------------------------------------------ remote backend
if ($Remote) {
    $Remote = $Remote.Trim().TrimEnd("/")
    if ($Remote -notmatch '^https://[A-Za-z0-9.-]+(:\d+)?$') { throw "-Remote must be an https:// address like https://name.trycloudflare.com" }
    if (-not $Key) { $Key = $env:UNMIX_BACKEND_KEY }
    if (-not $Key) {
        # Prompted rather than typed on the command line, so it stays out of shell history.
        $secure = Read-Host "Access key (printed by the notebook)" -AsSecureString
        $Key = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
    }
    try {
        $health = Invoke-RestMethod -Uri "$Remote/api/health" -Headers @{ "X-Unmix-Key" = $Key } -TimeoutSec 30
    } catch {
        $code = $_.Exception.Response.StatusCode.value__
        if ($code -eq 401) { throw "The remote backend refused the access key." }
        throw "Could not reach $Remote ($($_.Exception.Message)). Is the notebook still running?"
    }
    Write-Host "Remote backend: $($health.accelerator.summary), models $(if ($health.models_ready) { 'ready' } else { 'still loading' })" -ForegroundColor Green
    # Only the local proxy (Node) holds the key; the browser never sees it.
    $env:UNMIX_BACKEND_URL = $Remote
    $env:UNMIX_BACKEND_KEY = $Key
    try { Start-Frontend } finally { Remove-Item Env:UNMIX_BACKEND_KEY, Env:UNMIX_BACKEND_URL -ErrorAction SilentlyContinue }
    return
}

# ------------------------------------------------------------- local backend
if ($Fake) { $env:UNMIX_FAKE_MODELS = "1" } else { Remove-Item Env:UNMIX_FAKE_MODELS -ErrorAction SilentlyContinue }
Remove-Item Env:UNMIX_BACKEND_URL, Env:UNMIX_BACKEND_KEY -ErrorAction SilentlyContinue

if ($Accel) { $env:UNMIX_ACCEL = $Accel }
$pick = (uv run --no-project --quiet python (Join-Path $root "backend\scripts\pick_accel.py")) | ConvertFrom-Json
$group = $pick.group
Write-Host "Accelerator build: $group ($($pick.reason))" -ForegroundColor Cyan

# The voice-splitting environment (optional) gets the matching torch build.
$speakers = Join-Path $root "backend\speakers"
if (Test-Path (Join-Path $speakers ".venv")) {
    # Only the CUDA 13 build has a GPU variant there (see backend/speakers/pyproject.toml).
    $sgroup = if ($group -eq "cuda") { "cuda" } else { "cpu" }
    Push-Location $speakers
    try { uv sync --quiet --no-default-groups --group $sgroup } finally { Pop-Location }
}

Write-Host "Starting backend on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
$backend = Start-Process -FilePath "uv" -ArgumentList "run", "--no-default-groups", "--group", $group, "uvicorn", "app.server:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory (Join-Path $root "backend") -NoNewWindow -PassThru

try {
    Start-Frontend
}
finally {
    if ($backend -and -not $backend.HasExited) {
        Write-Host "Stopping backend..." -ForegroundColor Cyan
        # /T also ends the python child that uv spawned.
        taskkill /PID $backend.Id /T /F | Out-Null
    }
}
