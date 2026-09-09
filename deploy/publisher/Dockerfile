# escape=`
# No default is intentional: the operator must provide an approved immutable
# Windows CPython 3.11 image reference including @sha256:<digest>.
ARG PFC_PUBLISHER_BASE_IMAGE

FROM ${PFC_PUBLISHER_BASE_IMAGE} AS bundle
ARG PFC_PUBLISHER_BASE_IMAGE
SHELL ["powershell", "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"]
WORKDIR C:\src

# The ordinary context is positive-listed by the root .dockerignore. Wheels are
# staged under the single positive-listed build directory and are never fetched
# by this Dockerfile. This remains compatible with the Windows legacy builder.
COPY uv.lock C:\src\uv.lock
COPY deploy\publisher\environment-contract.json C:\src\deploy\publisher\environment-contract.json
COPY deploy\publisher\runtime-contract.json C:\src\deploy\publisher\runtime-contract.json
COPY deploy\publisher\operations-contract.json C:\src\deploy\publisher\operations-contract.json
COPY pfc_shaping C:\src\pfc_shaping
COPY scripts C:\src\scripts
COPY build\publisher-container-wheelhouse C:\wheelhouse
COPY build\publisher-container-wheelhouse-manifest.json C:\src\publisher-container-wheelhouse-manifest.json

RUN $ErrorActionPreference = 'Stop'; `
    New-Item -ItemType Directory -Path C:\bundle | Out-Null; `
    Copy-Item -LiteralPath C:\src\deploy\publisher\operations-contract.json -Destination C:\bundle\operations-contract.json; `
    Copy-Item -LiteralPath C:\src\publisher-container-wheelhouse-manifest.json -Destination C:\bundle\publisher-container-wheelhouse-manifest.json; `
    python.exe -m scripts.build_snapshot_publisher_runtime_closure `
      --wheel-directory C:\wheelhouse `
      --output C:\bundle\publisher-site-packages `
      --receipt-output C:\bundle\publisher-dependency-closure.json; `
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; `
    python.exe -m scripts.build_snapshot_publisher_zipapp build `
      --dependency-root C:\bundle\publisher-site-packages `
      --dependency-receipt C:\bundle\publisher-dependency-closure.json `
      --wheel-directory C:\wheelhouse `
      --output C:\bundle\fmv-lt-snapshot-publisher.pyz; `
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; `
    New-Item -ItemType Directory -Path C:\repro | Out-Null; `
    python.exe -m scripts.build_snapshot_publisher_runtime_closure `
      --wheel-directory C:\wheelhouse `
      --output C:\repro\publisher-site-packages `
      --receipt-output C:\repro\publisher-dependency-closure.json; `
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; `
    python.exe -m scripts.build_snapshot_publisher_zipapp build `
      --dependency-root C:\repro\publisher-site-packages `
      --dependency-receipt C:\repro\publisher-dependency-closure.json `
      --wheel-directory C:\wheelhouse `
      --output C:\repro\fmv-lt-snapshot-publisher.pyz; `
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; `
    if ((Get-FileHash C:\bundle\publisher-dependency-closure.json -Algorithm SHA256).Hash -cne (Get-FileHash C:\repro\publisher-dependency-closure.json -Algorithm SHA256).Hash) { throw 'dependency closure receipt is not reproducible' }; `
    if ((Get-FileHash C:\bundle\fmv-lt-snapshot-publisher.pyz -Algorithm SHA256).Hash -cne (Get-FileHash C:\repro\fmv-lt-snapshot-publisher.pyz -Algorithm SHA256).Hash) { throw 'publisher zipapp is not reproducible' }; `
    Remove-Item -LiteralPath C:\repro -Recurse -Force -ErrorAction Stop; `
    python.exe -m scripts.check_snapshot_publisher_container_context `
      --runtime-context C:\bundle `
      --base-image $env:PFC_PUBLISHER_BASE_IMAGE `
      --output C:\bundle\container-build-manifest.json; `
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

FROM ${PFC_PUBLISHER_BASE_IMAGE} AS runtime
ARG PFC_PUBLISHER_BASE_IMAGE
SHELL ["powershell", "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"]

LABEL org.opencontainers.image.title="FMV LT snapshot publisher" `
      org.opencontainers.image.base.name="${PFC_PUBLISHER_BASE_IMAGE}" `
      org.opencontainers.image.vendor="FMV SA" `
      org.opencontainers.image.licenses="LicenseRef-Proprietary" `
      org.fmv.pfc.job-kind="one-shot" `
      org.fmv.pfc.production-authorization="false"

COPY --from=bundle C:\bundle C:\pfc\runtime

RUN $ErrorActionPreference = 'Stop'; `
    New-Item -ItemType Directory -Path C:\pfc\scratch | Out-Null; `
    New-Item -ItemType Directory -Path C:\pfc\data | Out-Null; `
    New-Item -ItemType Directory -Path C:\pfc\evidence | Out-Null; `
    New-Item -ItemType Directory -Path C:\pfc\input | Out-Null; `
    $null = & icacls C:\pfc\runtime /inheritance:r /grant:r 'ContainerAdministrator:(OI)(CI)(F)' 'ContainerUser:(OI)(CI)(RX)'; `
    if ($LASTEXITCODE -ne 0) { throw 'runtime ACL admission failed' }; `
    $null = & icacls C:\pfc\scratch /inheritance:r /grant:r 'ContainerAdministrator:(OI)(CI)(F)' 'ContainerUser:(OI)(CI)(M)'; `
    if ($LASTEXITCODE -ne 0) { throw 'scratch ACL admission failed' }; `
    $null = & icacls C:\pfc\data /inheritance:r /grant:r 'ContainerAdministrator:(OI)(CI)(F)' 'ContainerUser:(OI)(CI)(M)'; `
    if ($LASTEXITCODE -ne 0) { throw 'data ACL admission failed' }; `
    $null = & icacls C:\pfc\evidence /inheritance:r /grant:r 'ContainerAdministrator:(OI)(CI)(F)' 'ContainerUser:(OI)(CI)(M)'; `
    if ($LASTEXITCODE -ne 0) { throw 'evidence ACL admission failed' }; `
    $null = & icacls C:\pfc\input /inheritance:r /grant:r 'ContainerAdministrator:(OI)(CI)(F)' 'ContainerUser:(OI)(CI)(RX)'; `
    if ($LASTEXITCODE -ne 0) { throw 'input ACL admission failed' }; `
    foreach ($path in @('C:\pfc\runtime','C:\pfc\scratch','C:\pfc\data','C:\pfc\evidence','C:\pfc\input')) { `
      $null = & icacls $path /verify; `
      if ($LASTEXITCODE -ne 0) { throw "noncanonical ACL on $path" } `
    }

ENV PFC_SNAPSHOT_PUBLISHER_DEPENDENCY_ROOT="C:\pfc\runtime\publisher-site-packages" `
    PFC_SNAPSHOT_PUBLISHER_SCRATCH_ROOT="C:\pfc\scratch" `
    PFC_SNAPSHOT_PUBLISHER_WORKER_TIMEOUT_SECONDS="900" `
    PFC_SNAPSHOT_PUBLISHER_ADMISSION_SLO_SECONDS="300" `
    TEMP="C:\pfc\scratch" `
    TMP="C:\pfc\scratch"

WORKDIR C:\pfc
USER ContainerUser

# This is a one-shot governed job, not a daemon. Exit codes and its structured
# stdout are the health signal; a periodic Docker healthcheck would be false.
HEALTHCHECK NONE

RUN $ErrorActionPreference = 'Stop'; `
    $runtimeProbe = 'C:\pfc\runtime\.write-probe'; `
    $inputProbe = 'C:\pfc\input\.write-probe'; `
    foreach ($probe in @($runtimeProbe, $inputProbe)) { `
      $writeSucceeded = $true; `
      try { Set-Content -LiteralPath $probe -Value 'forbidden' -ErrorAction Stop } catch { $writeSucceeded = $false }; `
      if ($writeSucceeded -or (Test-Path -LiteralPath $probe)) { throw "write unexpectedly admitted: $probe" } `
    }; `
    foreach ($probe in @('C:\pfc\scratch\.write-probe','C:\pfc\data\.write-probe','C:\pfc\evidence\.write-probe')) { `
      Set-Content -LiteralPath $probe -Value 'admitted' -ErrorAction Stop; `
      Remove-Item -LiteralPath $probe -Force -ErrorAction Stop `
    }; `
    & python.exe -I -S -B C:\pfc\runtime\fmv-lt-snapshot-publisher.pyz --help | Out-Null; `
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; `
    & python.exe -I -S -B C:\pfc\runtime\fmv-lt-snapshot-publisher.pyz prepare --help | Out-Null; `
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; `
    & python.exe -I -S -B C:\pfc\runtime\fmv-lt-snapshot-publisher.pyz cas --help | Out-Null; `
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; `
    & python.exe -I -S -B C:\pfc\runtime\fmv-lt-snapshot-publisher.pyz finalize --help | Out-Null; `
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

CMD []
ENTRYPOINT ["python.exe", "-I", "-S", "-B", "C:\\pfc\\runtime\\fmv-lt-snapshot-publisher.pyz"]
