$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'mujoco_hotel\view_hotel.py')
exit $LASTEXITCODE
