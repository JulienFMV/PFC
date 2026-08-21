"""Positive inventory for the isolated LT snapshot publisher zipapp."""

PUBLISHER_ARTIFACT_SCHEMA = "fmv_lt_snapshot_publisher_artifact.v2"
PUBLISHER_ARTIFACT_TIMESTAMP = (2026, 7, 14, 0, 0, 0)

PUBLISHER_RUNTIME_PYTHON_FILES = frozenset(
    """
pfc_shaping/__init__.py
pfc_shaping/durable_artifact.py
pfc_shaping/exit_codes.py
pfc_shaping/parquet_safety.py
pfc_shaping/path_safety.py
pfc_shaping/publisher_package_contract.py
pfc_shaping/publisher_runtime_admission.py
pfc_shaping/data/__init__.py
pfc_shaping/data/acquisition_contract.py
pfc_shaping/data/governed_lt_acquisition.py
pfc_shaping/data/lt_input_replay.py
pfc_shaping/data/lt_input_sources.py
pfc_shaping/data/lt_replay_transforms.py
pfc_shaping/data/shared_data_root.py
pfc_shaping/data/snapshot_anchor_client.py
pfc_shaping/data/snapshot_publication_contract.py
pfc_shaping/data/snapshot_publication_state.py
pfc_shaping/data/snapshot_publisher.py
pfc_shaping/pipeline/__init__.py
pfc_shaping/pipeline/strict_structured_data.py
scripts/publish_governed_lt_input_snapshot.py
""".split()
)

PUBLISHER_SYNTHETIC_FILES = frozenset({"__main__.py", "scripts/__init__.py"})
PUBLISHER_MANIFEST_PATH = "FMV-SNAPSHOT-PUBLISHER-MANIFEST.json"
PUBLISHER_ENVIRONMENT_CONTRACT_PATH = (
    "FMV-SNAPSHOT-PUBLISHER-ENVIRONMENT-CONTRACT.json"
)
PUBLISHER_RUNTIME_CONTRACT_PATH = "FMV-SNAPSHOT-PUBLISHER-RUNTIME-CONTRACT.json"
PUBLISHER_EMBEDDED_CONTRACT_FILES = {
    PUBLISHER_ENVIRONMENT_CONTRACT_PATH: "deploy/publisher/environment-contract.json",
    PUBLISHER_RUNTIME_CONTRACT_PATH: "deploy/publisher/runtime-contract.json",
}

PUBLISHER_FORBIDDEN_FILES = frozenset(
    {
        "pfc_shaping/data/acquisition_signing.py",
        "pfc_shaping/data/snapshot_anchor_signing.py",
        "pfc_shaping/data/snapshot_anchor_reference.py",
        "pfc_shaping/data/snapshot_bootstrap_authority.py",
        "pfc_shaping/data/snapshot_legacy_publication_signing.py",
        "scripts/sign_snapshot_bootstrap_authorization.py",
    }
)
