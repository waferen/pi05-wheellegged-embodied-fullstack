$ErrorActionPreference = 'Stop'
$taskStatusPath = Join-Path $PSScriptRoot 'setup_job_status.json'
$taskLogPath = Join-Path $PSScriptRoot 'setup_job.log'
function Write-TaskStatus([string]$phase, [string]$detail) {
    @{ phase=$phase; detail=$detail; pid=$PID; updated_at=(Get-Date).ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath $taskStatusPath -Encoding utf8
}
Start-Transcript -LiteralPath $taskLogPath -Append | Out-Null
try {
    Write-TaskStatus 'running' 'Import Ubuntu, prepare dependencies, check CUDA and attempt the local LoRA probe.'
    & (Join-Path $PSScriptRoot 'Resume-WslSetup.ps1') -RunProbe
    Write-TaskStatus 'complete' 'Bootstrap and probe commands completed; inspect GPU and probe result JSON for actual evidence.'
} catch {
    Write-TaskStatus 'failed' $_.Exception.Message
    Write-Error $_ -ErrorAction Continue
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
