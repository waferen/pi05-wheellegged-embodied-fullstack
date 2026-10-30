param(
    [ValidateRange(1, 65535)][int]$Port = 8765,
    [string]$HostName = '127.0.0.1',
    [ValidateRange(1, 50)][int]$ExecuteSteps = 25,
    [ValidateSet('Follow', 'Overview', 'Head', 'Hand')][string]$View = 'Follow'
)
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$taskViewer = Join-Path $PSScriptRoot 'mujoco_hotel/view_pi05.py'
Write-Host "Connecting to PI0.5 at ${HostName}:$Port. If absent, first run competition_local/serve_pi05.ps1 -Port $Port in another terminal."
& $taskPython $taskViewer --host $HostName --port $Port --execute-steps $ExecuteSteps --view $View
exit $LASTEXITCODE
