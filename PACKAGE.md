# FMV PFC LT

Governed Swiss long-term power forward curve runtime.

The package deliberately installs no console-script executable. Local governed
operations use the launcherless module route
`python.exe -I -B -m pfc_shaping.cli.governed_release`, with the exact runtime
receipt path and its caller-held SHA-256 supplied through the documented
runtime environment. Build, finalize, register, audit, and status remain
phase-separated. Promote and rollback are hard-disabled for this local runtime
and require a future independently signed, IT-admitted production runtime
attestation. Runtime data, configuration, trust anchors, and secrets are
supplied through explicit external mounts; none are embedded in the
distribution.
