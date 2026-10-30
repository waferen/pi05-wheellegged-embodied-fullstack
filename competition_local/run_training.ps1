param([int]$Steps = 65536)
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run setup_windows.ps1 first.' }
& $taskPython (Join-Path $PSScriptRoot 'mujoco_training\build_scene.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $taskPython (Join-Path $PSScriptRoot 'mujoco_training\train_smoke.py') --steps $Steps
exit $LASTEXITCODE
