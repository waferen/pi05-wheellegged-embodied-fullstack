param(
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$RepoId = 'local/mujoco_b2w_piper',
    [switch]$AllowUnreviewed,
    [switch]$AllowSmoke
)
$ErrorActionPreference = 'Stop'
function Convert-TaskLinuxPath([string]$TaskPath) {
    $taskAbsolute = [IO.Path]::GetFullPath($TaskPath)
    if ($taskAbsolute -notmatch '^([A-Za-z]):\\') { throw 'Expected a local drive path' }
    return '/mnt/' + $taskAbsolute.Substring(0,1).ToLowerInvariant() + '/' + $taskAbsolute.Substring(3).Replace('\','/')
}
$taskScript = Convert-TaskLinuxPath (Join-Path $PSScriptRoot 'mujoco_hotel/convert_mujoco_to_lerobot.py')
$taskArguments = @('--distribution','PPSUC-Ubuntu22','--user','eai','--exec','/home/eai/ppsuc/.venv/bin/python',
    $taskScript,'--source',(Convert-TaskLinuxPath $Source),'--output',(Convert-TaskLinuxPath $Output),'--repo-id',$RepoId)
if ($AllowUnreviewed) { $taskArguments += '--allow-unreviewed' }
if ($AllowSmoke) { $taskArguments += '--allow-smoke' }
& wsl.exe @taskArguments
exit $LASTEXITCODE
