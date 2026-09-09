param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$Csv = ""
)

$ErrorActionPreference = "Stop"
$pbip = Join-Path $RepoRoot "powerbi\PFC_QA.pbip"

if (-not (Test-Path -LiteralPath $pbip)) {
    throw "Cannot find $pbip"
}

& (Join-Path $RepoRoot "powerbi\refresh_powerbi_data.ps1") -RepoRoot $RepoRoot -Csv $Csv

Push-Location $RepoRoot
try {
    python (Join-Path $RepoRoot "scripts\build_powerbi_semantic_model.py") --no-backup
    python (Join-Path $RepoRoot "scripts\build_powerbi_report_layout.py") --no-backup
}
finally {
    Pop-Location
}

Invoke-Item -LiteralPath $pbip
