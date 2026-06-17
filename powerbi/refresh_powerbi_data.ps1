param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$Csv = ""
)

$ErrorActionPreference = "Stop"
$script = Join-Path $RepoRoot "scripts\build_powerbi_exports.py"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Cannot find $script"
}

Push-Location $RepoRoot
try {
    if ([string]::IsNullOrWhiteSpace($Csv)) {
        python $script --output-dir "powerbi\data"
    }
    else {
        python $script --csv $Csv --output-dir "powerbi\data"
    }
}
finally {
    Pop-Location
}

Write-Host "[powerbi] data refreshed in $RepoRoot\powerbi\data"
