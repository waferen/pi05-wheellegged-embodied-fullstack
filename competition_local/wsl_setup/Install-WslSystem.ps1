$ErrorActionPreference = 'Stop'
$taskDirectory = $PSScriptRoot
$taskStatusPath = Join-Path $taskDirectory 'system_install_status.json'
function Write-TaskStatus([hashtable]$Fields) {
    $Fields['updated_at'] = (Get-Date).ToString('o')
    $Fields | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $taskStatusPath -Encoding utf8
}
$taskTranscript = Join-Path $taskDirectory 'system_install_transcript.log'
Start-Transcript -LiteralPath $taskTranscript -Append | Out-Null
try {
    $taskIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $taskPrincipal = [Security.Principal.WindowsPrincipal]::new($taskIdentity)
    if (-not $taskPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Administrator authorization is required for Windows optional features.' }
    Write-TaskStatus @{ phase = 'checking_features'; administrator = $true; automatic_restart = $false }
    $taskFeatureBefore = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
    $taskRestartNeeded = $false
    if ($taskFeatureBefore.State -ne 'Enabled') {
        Write-TaskStatus @{ phase = 'enabling_virtual_machine_platform'; previous_state = [string]$taskFeatureBefore.State; automatic_restart = $false }
        $taskFeatureResult = Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -All -NoRestart
        $taskRestartNeeded = [bool]$taskFeatureResult.RestartNeeded
    }
    Write-TaskStatus @{ phase = 'installing_wsl_package'; restart_required = $taskRestartNeeded; automatic_restart = $false }
    & "$env:WINDIR\System32\wsl.exe" --install --no-distribution --web-download
    $taskWslExit = $LASTEXITCODE
    if ($taskWslExit -notin @(0, 3010)) { throw "WSL installation returned exit code $taskWslExit. See transcript." }
    if ($taskWslExit -eq 3010) { $taskRestartNeeded = $true }
    $taskFeatureAfter = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
    Write-TaskStatus @{ phase = 'system_install_finished'; wsl_exit_code = $taskWslExit; virtual_machine_platform = [string]$taskFeatureAfter.State; restart_required = $taskRestartNeeded; automatic_restart = $false }
} catch {
    Write-TaskStatus @{ phase = 'failed'; error = $_.Exception.Message; automatic_restart = $false }
    throw
} finally { Stop-Transcript | Out-Null }
