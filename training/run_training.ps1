# Overnight training of the road obstacle detector, with live progress in this terminal.
# Runs training/train_road_obstacles.ipynb inside the GPU training container (papermill): every line the
# notebook prints appears here live, and the executed notebook (outputs, charts, metrics) is saved in place.
# A copy of the output goes to training/runs/train.log.
#
#   pwsh -ExecutionPolicy Bypass -File training/run_training.ps1               # 8 h of training
#   pwsh -ExecutionPolicy Bypass -File training/run_training.ps1 -Hours 10     # longer
#   pwsh -ExecutionPolicy Bypass -File training/run_training.ps1 -Resume       # continue an interrupted night
#
# Keep the laptop plugged in with the lid open: this script stops Windows from sleeping while it runs.

param([double]$Hours = 8, [switch]$Resume)

$ErrorActionPreference = "Stop"
# Docker prints UTF-8 (progress bars, arrows): read and show it as UTF-8, not the old console code page.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$project = Split-Path -Parent $PSScriptRoot
Set-Location $project
$Host.UI.RawUI.WindowTitle = "SmartCar - overnight training of the road obstacle detector"
New-Item -ItemType Directory -Force training/runs | Out-Null

# Keep the PC awake (the screen may still turn off) until this script ends.
Add-Type -Namespace Win32 -Name Power -MemberDefinition `
    '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null

$finish = (Get-Date).AddHours($Hours + 0.5).ToString("HH:mm")
Write-Host ""
Write-Host "  Road obstacle detector - overnight training" -ForegroundColor Cyan
Write-Host "  Notebook : training/train_road_obstacles.ipynb (outputs are saved into it)"
Write-Host "  Log      : training/runs/train.log"
Write-Host "  GPU      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
Write-Host "  Budget   : $Hours h of training  ->  expected end around $finish$(if ($Resume) { '  (resuming)' })"
Write-Host "  The PC will not go to sleep while this window runs. Keep it plugged in, lid open."
Write-Host ""

Write-Host "  Stopping the dashboard during training (it is restarted at the end)..." -ForegroundColor DarkGray
docker compose stop dashboard 2>&1 | Out-Null

$params = @("-p", "TRAIN_HOURS", "$Hours")
if ($Resume) { $params += @("-p", "RESUME", "True") }

try {
    docker compose --profile train run --rm trainer `
        papermill train_road_obstacles.ipynb train_road_obstacles.ipynb @params --log-output --progress-bar 2>&1 |
        Tee-Object -FilePath training/runs/train.log
    $ok = ($LASTEXITCODE -eq 0)
}
finally {
    [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
}

Write-Host ""
docker compose up -d dashboard 2>&1 | Out-Null
docker compose restart dashboard 2>&1 | Out-Null
if ($ok) {
    Write-Host "  Training finished - the dashboard now uses the new models." -ForegroundColor Green
    Write-Host "  Open http://localhost:8050 -> Real-world -> Camera / AI models" -ForegroundColor Green
} else {
    Write-Host "  Training stopped with an error - see above or training/runs/train.log." -ForegroundColor Red
    Write-Host "  If it was interrupted during training, run again with -Resume." -ForegroundColor Red
}
