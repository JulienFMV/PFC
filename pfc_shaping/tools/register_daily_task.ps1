param(
    [string]$TaskName = "PFC_Daily_Update",
    [string]$RunAt = "06:15",
    [string]$PythonExe = "C:\Users\jbattaglia\.conda\ppa_env\python.exe",
    [string]$RepoRoot = "H:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC"
)

$ErrorActionPreference = "Stop"

throw "Registration of the legacy direct publisher is disabled. Deploy the governed runner instead."

$scriptPath = Join-Path $RepoRoot "pfc_shaping\tools\run_daily_pfc.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -PythonExe `"$PythonExe`" -RepoRoot `"$RepoRoot`""
$trigger = New-ScheduledTaskTrigger -Daily -At $RunAt
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Force
Write-Host "Task '$TaskName' registered at $RunAt"
