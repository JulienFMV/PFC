param(
    [string]$PythonExe = "C:\Users\jbattaglia\.conda\pfc311\python.exe",
    [string]$RepoRoot = "H:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC"
)

$ErrorActionPreference = "Stop"

Set-Location $RepoRoot
& $PythonExe -m pfc_shaping.pipeline.rolling_update
if ($LASTEXITCODE -ne 0) {
    throw "PFC pipeline failed with exit code $LASTEXITCODE"
}
