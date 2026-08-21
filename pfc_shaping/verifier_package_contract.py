"""Positive inventory for the isolated provider-evidence verifier zipapp."""

VERIFIER_ARTIFACT_SCHEMA = "fmv_lt_provider_verifier_artifact.v1"
VERIFIER_ARTIFACT_TIMESTAMP = (2026, 7, 24, 0, 0, 0)

VERIFIER_RUNTIME_PYTHON_FILES = frozenset(
    """
pfc_shaping/__init__.py
pfc_shaping/build_identity.py
pfc_shaping/durable_artifact.py
pfc_shaping/exit_codes.py
pfc_shaping/parquet_safety.py
pfc_shaping/path_safety.py
pfc_shaping/publisher_runtime_admission.py
pfc_shaping/verifier_package_contract.py
pfc_shaping/verifier_runtime_admission.py
pfc_shaping/cli/__init__.py
pfc_shaping/cli/audit_legacy_provider_resolution.py
pfc_shaping/cli/audit_provider_acquisition_quarantine.py
pfc_shaping/data/__init__.py
pfc_shaping/data/acquisition_contract.py
pfc_shaping/data/eex_forward_local_capture.py
pfc_shaping/data/eex_forward_vintage_intake.py
pfc_shaping/data/eex_historical_vintage.py
pfc_shaping/data/governed_lt_acquisition.py
pfc_shaping/data/ingest_forwards.py
pfc_shaping/pipeline/__init__.py
pfc_shaping/pipeline/strict_structured_data.py
scripts/audit_ch_eex_current_local_capture.py
""".split()
)

VERIFIER_SYNTHETIC_FILES = frozenset({"__main__.py", "scripts/__init__.py"})
VERIFIER_MANIFEST_PATH = "FMV-LT-PROVIDER-VERIFIER-MANIFEST.json"
VERIFIER_RUNTIME_CONTRACT_PATH = "FMV-LT-PROVIDER-VERIFIER-RUNTIME-CONTRACT.json"
VERIFIER_EMBEDDED_CONTRACT_FILES = {
    VERIFIER_RUNTIME_CONTRACT_PATH: "deploy/verifier/runtime-contract.json",
}
VERIFIER_COMMANDS = frozenset(
    {
        "audit-acquisition",
        "audit-current-eex",
        "audit-legacy-resolution",
        "runtime-check",
    }
)

# The dated EEX selection binds these exact parser sources.  The verifier
# builder refuses to package a parser that differs from the selected evidence.
VERIFIER_FROZEN_SOURCE_HASHES = {
    "pfc_shaping/data/eex_forward_vintage_intake.py": (
        "0cde82c272705d90eb8f23837c96cb2769cf2885fe2549618e759e987d3e8854"
    ),
    "pfc_shaping/data/ingest_forwards.py": (
        "39ec58287abaf945e4d5537190057cb0592e8a177f8d6f3197d9f8d5fd25e42a"
    ),
}

VERIFIER_FORBIDDEN_FILES = frozenset(
    {
        "pfc_shaping/data/acquisition_signing.py",
        "pfc_shaping/data/snapshot_anchor_signing.py",
        "pfc_shaping/data/snapshot_bootstrap_authority.py",
        "pfc_shaping/pipeline/atomic_promotion.py",
        "pfc_shaping/pipeline/candidate_finalize.py",
        "scripts/publish_governed_lt_input_snapshot.py",
    }
)
