param(
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [switch]$Preflight
)
$ErrorActionPreference = 'Stop'
$linuxScript = '/mnt/d/EAI Competition Of PPSUC/competition_local/serve_pi05.py'
$arguments = @('--distribution', 'PPSUC-Ubuntu22', '--user', 'eai', '--exec', '/home/eai/ppsuc/.venv/bin/python', $linuxScript, '--port', "$Port")
if ($Preflight) { $arguments += '--preflight' }
& wsl.exe @arguments
exit $LASTEXITCODE
