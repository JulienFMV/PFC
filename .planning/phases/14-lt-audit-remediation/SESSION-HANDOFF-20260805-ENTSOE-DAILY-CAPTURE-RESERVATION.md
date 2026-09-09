# Session handoff - ENTSO-E daily capture reservation

Date: 2026-08-05  
Decision: `D-20260805-238`

## Outcome

D238 implements a local fail-closed structural guard for the user's maximum
one ENTSO-E repatriation per `Europe/Zurich` day. It binds the exact D233 daily
ceiling, D235 external-time proof and D236 bounded-cost proposal evidence.

The guard:

- derives the local day from the embedded D236 proposal creation timestamp;
- allows at most one reservation entry per Zurich day;
- requires unique reservation, capture-batch and proposal IDs;
- requires strictly increasing reservation timestamps;
- counts reservation, execution start, success and failure as consuming the
  day, preventing a same-day retry after a costly failure;
- preserves the exact D236 cost fence and leaves all authority false.

## Files and hashes

- contract: `.planning/phases/14-lt-audit-remediation/ENTSOE-DAILY-CAPTURE-RESERVATION-LEDGER-CONTRACT-V1.json`
  - raw SHA-256 `827efab999bd7bb9446f753b4b98be57f2638ff4673481d1f3683458084be251`
  - content ID `18bf70f82ae3ab078660b765cf131369757002c056a96873fd66e93ded959544`
- validator: `pfc_shaping/validation/entsoe_daily_capture_reservation_ledger.py`
  - SHA-256 `d8a932d2f813cbf457d7906234c29c4e84c8f010f6486ef197d26a2a05efc69a`
- tests: `tests/test_entsoe_daily_capture_reservation_ledger.py`
  - SHA-256 `2a1074f95e1d86bfaf0742008e8017593e90a9b2a7389f86480824e9d559ae95`
- materializer: `build/databricks-eex-daily/materialize_entsoe_daily_capture_reservation_ledger.py`
  - SHA-256 `0f9a30b3e851757c845dfb5fcdd69dafc679c24e47b8f3609f8ed6cb1476cb38`

## Proof and verification

- proof content ID:
  `2b9f6c513e0382e685bee78fb02d6c071a6954a26e715c5784fb28efec878aa8`
- proof path:
  `build/databricks-eex-daily/2026-08-05/entsoe-daily-capture-reservation-ledger-proofs/2b9f6c513e0382e685bee78fb02d6c071a6954a26e715c5784fb28efec878aa8/`
- manifest SHA-256:
  `922a31ce0bf54b28e00b352bae1e4fa1d66fa357adf687c9a5f456e94d4511f6`
- assessment SHA-256:
  `264abefb49b92a06a77c1dfda3d9389f263a0a4771c73d86d4abb0b3b532d671`
- focused: `20 passed`;
- adjacent D233-D238: `261 passed`;
- Ruff: passed;
- two materializations: identical proof content ID.

No Databricks connection or statement, Warehouse start, network call, `H:`
access, remote write, market value, ENTSO-E value or AFRY value occurred.

## Limits and next step

This is structural synthetic evidence only. D238 deliberately does not
implement a real ledger, authorization receipt admission, atomic persistence,
cross-process locking or execution. Those require a separately approved exact
D236 proposal and explicit maximum cost. Real metadata/mapping/values, data
quality, PIT and an independent future holdout remain absent. Training,
selection, model input, candidate assembly, promotion and production remain
false; T057 stays sealed.
