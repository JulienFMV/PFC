[CmdletBinding()]
param(
    [switch]$Execute,
    [switch]$IncludeOutputs
)

$ErrorActionPreference = 'Stop'
$expectedRoot = [IO.Path]::GetFullPath('C:\Users\jbattaglia\PFC_LT').TrimEnd('\')
$currentRoot = [IO.Path]::GetFullPath((Get-Location).Path).TrimEnd('\')
$gitRoot = [IO.Path]::GetFullPath(
    ((git rev-parse --show-toplevel).Trim() -replace '/', '\')
).TrimEnd('\')

if ($currentRoot -ne $expectedRoot -or $gitRoot -ne $expectedRoot) {
    throw "Run only from the canonical workspace: $expectedRoot"
}

$candidates = @(
    Get-ChildItem -Force -Directory | Where-Object {
        $_.Name -eq '__pycache__' -or
        $_.Name -eq '.pytest_cache' -or
        $_.Name -eq '.ruff_cache' -or
        $_.Name -eq 'fmv_pfc_lt.egg-info' -or
        $_.Name -like 'pytest-*' -or
        $_.Name -like '.pytest-*' -or
        $_.Name -like '.tmp_pytest*'
    }
)
$allowedParents = @($expectedRoot)

if ($IncludeOutputs) {
    $outputRoots = @(
        [IO.Path]::GetFullPath((Join-Path $expectedRoot 'output')).TrimEnd('\'),
        [IO.Path]::GetFullPath(
            (Join-Path $expectedRoot 'pfc_shaping\output')
        ).TrimEnd('\')
    )
    foreach ($outputRoot in $outputRoots) {
        if (
            -not $outputRoot.StartsWith(
                $expectedRoot + '\',
                [StringComparison]::OrdinalIgnoreCase
            ) -or
            -not (Test-Path -LiteralPath $outputRoot -PathType Container) -or
            ((Get-Item -LiteralPath $outputRoot -Force).Attributes -band
                [IO.FileAttributes]::ReparsePoint)
        ) {
            throw "Unsafe generated-output root: $outputRoot"
        }
        $allowedParents += $outputRoot
        $candidates += @(
            Get-ChildItem -LiteralPath $outputRoot -Force |
                Where-Object { $_.Name -ne '.gitkeep' }
        )
    }
}

$rows = @()
$totalBytes = 0L
$totalFiles = 0L
foreach ($candidate in $candidates) {
    $fullPath = [IO.Path]::GetFullPath($candidate.FullName).TrimEnd('\')
    $parentPath = [IO.Path]::GetFullPath(
        [IO.Path]::GetDirectoryName($fullPath)
    ).TrimEnd('\')
    if (
        $parentPath -notin $allowedParents -or
        -not $fullPath.StartsWith(
            $expectedRoot + '\',
            [StringComparison]::OrdinalIgnoreCase
        ) -or
        ($candidate.Attributes -band [IO.FileAttributes]::ReparsePoint)
    ) {
        throw "Unsafe cleanup candidate: $fullPath"
    }
    $relativePath = $fullPath.Substring($expectedRoot.Length + 1) -replace '\\', '/'
    $trackedPaths = @(git ls-files -- $relativePath)
    if ($trackedPaths.Count -gt 0) {
        throw "Refusing to delete tracked path: $relativePath"
    }
    $files = if ($candidate.PSIsContainer) {
        @(Get-ChildItem -LiteralPath $fullPath -File -Recurse -Force)
    } else {
        @($candidate)
    }
    $bytes = ($files | Measure-Object Length -Sum).Sum
    $totalBytes += $bytes
    $totalFiles += $files.Count
    $rows += [pscustomobject]@{
        Name = $relativePath
        Files = $files.Count
        MiB = [math]::Round($bytes / 1MB, 1)
    }
}

foreach ($row in ($rows | Sort-Object MiB -Descending)) {
    Write-Output "candidate=$($row.Name);files=$($row.Files);mib=$($row.MiB)"
}
Write-Output "candidate_directories=$($candidates.Count)"
Write-Output "candidate_files=$totalFiles"
Write-Output "candidate_mib=$([math]::Round($totalBytes / 1MB, 1))"

if (-not $Execute) {
    Write-Output 'dry_run=true; rerun with -Execute and the same scope switches'
    exit 0
}

foreach ($candidate in $candidates) {
    Remove-Item -LiteralPath $candidate.FullName -Recurse -Force
}
Write-Output "removed_directories=$($candidates.Count)"
