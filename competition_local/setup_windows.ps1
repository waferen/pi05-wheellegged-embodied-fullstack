param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$taskTemp = Join-Path $PSScriptRoot 'temp'
New-Item -ItemType Directory -Force -Path $taskTemp | Out-Null
$env:TEMP = $taskTemp
$env:TMP = $taskTemp
& $Python -m venv (Join-Path $PSScriptRoot '.venv')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') -m pip install -r (Join-Path $PSScriptRoot 'requirements.lock.txt')
exit $LASTEXITCODE
