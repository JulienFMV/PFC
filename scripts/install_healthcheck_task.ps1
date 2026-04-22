param(
    [string]$TaskName = "PFC_Healthcheck_ShortTerm",
    [string]$StartTime = "06:15",
    [string]$PythonExe = "C:\Users\jbattaglia\.conda\ppa_env\python.exe"
)

$ErrorActionPreference = "Stop"

$RunScript = Join-Path $PSScriptRoot "run_healthcheck_short_term.ps1"
if (!(Test-Path -LiteralPath $RunScript)) {
    throw "Runner script not found: $RunScript"
}

if (-not (Get-Module -ListAvailable -Name ScheduledTasks)) {
    throw "ScheduledTasks module is not available on this system."
}

Write-Host "Creating/updating scheduled task '$TaskName' at $StartTime ..."
$actionArgs = '-NoProfile -ExecutionPolicy Bypass -File "' + $RunScript + '" -PythonExe "' + $PythonExe + '"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $actionArgs
$triggerTime = [datetime]::ParseExact($StartTime, "HH:mm", $null)
$trigger = New-ScheduledTaskTrigger -Daily -At $triggerTime
$userId = "$env:USERDOMAIN\$env:USERNAME"
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 4)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Daily short-term PFC healthcheck (AST + eval + optional production dry run)." `
    -Force | Out-Null

Write-Host ""
Write-Host "Scheduled task details:"
Get-ScheduledTask -TaskName $TaskName | Format-List TaskName,State,Author,Description,URI | Out-Host
