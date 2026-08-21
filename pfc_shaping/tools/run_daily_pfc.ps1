param(
    [string]$PythonExe = "C:\Users\jbattaglia\.conda\ppa_env\python.exe",
    [string]$RepoRoot = "H:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC"
)

$ErrorActionPreference = "Stop"

throw "Legacy rolling_update publication is disabled. Use the governed candidate/audit/receipt/promotion runner."

Set-Location $RepoRoot
$envRoot = Split-Path -Parent $PythonExe
$env:PATH = "$envRoot;$envRoot\Library\bin;$envRoot\Scripts;$env:PATH"
& $PythonExe -m pfc_shaping.pipeline.rolling_update
if ($LASTEXITCODE -ne 0) {
    throw "PFC pipeline failed with exit code $LASTEXITCODE"
}
