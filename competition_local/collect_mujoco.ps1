param(
    [string]$Output = (Join-Path $PSScriptRoot 'datasets/mujoco_raw'),
    [string]$OperatorName = 'local',
    [switch]$NoPanel,
    [switch]$SmokeTest
)
$ErrorActionPreference = 'Stop'
$taskArguments = @((Join-Path $PSScriptRoot 'mujoco_hotel/collect_mujoco.py'), '--output', $Output, '--operator', $OperatorName)
if ($NoPanel) { $taskArguments += '--no-panel' }
if ($SmokeTest) { $taskArguments += '--smoke-test' }
& (Join-Path $PSScriptRoot '.venv/Scripts/python.exe') @taskArguments
exit $LASTEXITCODE
