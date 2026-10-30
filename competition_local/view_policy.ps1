param([int]$Episodes = 10)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'mujoco_training\replay.py') --viewer --episodes $Episodes
exit $LASTEXITCODE
