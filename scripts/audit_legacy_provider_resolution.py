"""Fail-closed checkout stub; use the isolated provider verifier zipapp."""

from pfc_shaping.exit_codes import EXIT_INTEGRITY_FAILURE


def main() -> int:
    return EXIT_INTEGRITY_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
