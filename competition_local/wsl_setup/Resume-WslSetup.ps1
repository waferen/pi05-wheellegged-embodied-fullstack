param([switch]$RunProbe)
$ErrorActionPreference = 'Stop'
$taskDistro = 'PPSUC-Ubuntu22'
$taskImage = Join-Path $PSScriptRoot 'ubuntu-22.04.5-wsl-amd64.wsl'
$taskInstall = Join-Path $PSScriptRoot 'distro'
$taskRoot = (Split-Path -Parent $PSScriptRoot)
$taskLinuxRoot = '/mnt/d/EAI Competition Of PPSUC/competition_local'
foreach ($taskShell in @('prepare_linux.sh','run_lora_probe.sh','network_env.sh')) {
    $taskPath = Join-Path $PSScriptRoot $taskShell
    $taskText = [IO.File]::ReadAllText($taskPath).Replace("`r`n","`n")
    [IO.File]::WriteAllText($taskPath,$taskText,[Text.UTF8Encoding]::new($false))
}
$taskNames = (& wsl.exe --list --quiet 2>$null) -replace "`0",''
if ($taskDistro -notin $taskNames) {
    if ((Get-FileHash -LiteralPath $taskImage -Algorithm SHA256).Hash.ToLowerInvariant() -ne '4499c4fe257f2fc83145b429ce211a0a43fd590e70d6261ede616210947d9f8f') { throw 'Ubuntu image checksum mismatch' }
    & wsl.exe --import $taskDistro $taskInstall $taskImage --version 2
    if ($LASTEXITCODE -ne 0) { throw 'WSL2 import failed. If Windows component installation requested a reboot, reboot first; no distribution will be unregistered automatically.' }
}
$taskNetworkEnv = @()
$taskProxy = Get-ItemProperty -LiteralPath 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings'
if ($taskProxy.ProxyEnable -eq 1 -and $taskProxy.ProxyServer -match '^127\.0\.0\.1:([0-9]+)$') {
    $taskNetworkEnv = @("PPSUC_HOST_PROXY_PORT=$($Matches[1])")
}
& wsl.exe -d $taskDistro -u root -- env @taskNetworkEnv bash "$taskLinuxRoot/wsl_setup/prepare_linux.sh" $taskLinuxRoot
if ($LASTEXITCODE -ne 0) { throw 'Linux bootstrap failed; inspect linux_bootstrap.log.' }
if ($RunProbe) {
    & wsl.exe -d $taskDistro -u eai -- env @taskNetworkEnv bash "$taskLinuxRoot/wsl_setup/run_lora_probe.sh" $taskLinuxRoot 10
    if ($LASTEXITCODE -ne 0) { throw 'LoRA probe did not pass; inspect the exact output before changing the experiment.' }
}
