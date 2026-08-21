# Local Chronos-2 adapter metadata

This directory records metadata for an experimental short-term Chronos-2
adapter. Model weights are local build artifacts and are not distributed by
Git (`*.safetensors` is ignored).

The adapter is not part of the long-term PFC model, does not authorize a
production release and must not be used as a substitute for the governed LT
EEX/ENTSO-E evidence chain.

To reproduce the experiment, use `scripts/finetune_chronos2.py` with an
explicitly governed local base model and input snapshot. Keep the resulting
weights below a governed local artifact directory; record their hash and
provenance separately before any comparison.
