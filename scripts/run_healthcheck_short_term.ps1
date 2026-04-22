param(
    [string]$PythonExe = "C:\Users\jbattaglia\.conda\ppa_env\python.exe"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$OutputDir = Join-Path $RepoRoot "pfc_shaping\output"
$DateStamp = Get-Date -Format "yyyy-MM-dd"
$OutputPath = Join-Path $OutputDir "healthcheck_short_term_$DateStamp.json"
$LogPath = Join-Path $RepoRoot "healthcheck_short_term_$DateStamp.log"

if (!(Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$env:PATH = "C:\Users\jbattaglia\.conda\ppa_env;C:\Users\jbattaglia\.conda\ppa_env\Library\bin;C:\Users\jbattaglia\.conda\ppa_env\Scripts;" + $env:PATH
$env:PYTHONIOENCODING = "utf-8"

& $PythonExe (Join-Path $RepoRoot "scripts\healthcheck_short_term.py") `
    --with-production `
    --output $OutputPath `
    --log $LogPath

exit $LASTEXITCODE

