param(
    [Parameter(Mandatory=$true,Position=0)][ValidateSet('validate','review','replay')][string]$Operation,
    [Parameter(Mandatory=$true,Position=1)][string]$Path,
    [ValidateSet('success','failure','exclude')][string]$Result,
    [string]$Note = '',
    [switch]$Complete
)
$ErrorActionPreference = 'Stop'
$taskInputPath = (Resolve-Path -LiteralPath $Path).Path
$taskArguments = @('-m', 'collection.pipeline', $Operation, $taskInputPath)
if ($Result) { $taskArguments += @('--result', $Result, '--note', $Note) }
if ($Complete) { $taskArguments += '--complete' }
Push-Location (Join-Path $PSScriptRoot 'mujoco_hotel')
try { & (Join-Path $PSScriptRoot '.venv/Scripts/python.exe') @taskArguments; $taskExit = $LASTEXITCODE }
finally { Pop-Location }
exit $taskExit
