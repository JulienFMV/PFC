param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"
$script = Join-Path $RepoRoot "scripts\build_powerbi_exports.py"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Cannot find $script"
}

Push-Location $RepoRoot
try {
    python $script --output-dir "powerbi\data"
}
finally {
    Pop-Location
}

Write-Host "[powerbi] data refreshed in $RepoRoot\powerbi\data"
