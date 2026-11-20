Reference adapter provenance
============================

The canonical text producer is the Google Gemma repository at the immutable
commit in GOOGLE-PIN.json. `fetch_google.py --cache PATH` explicitly downloads
that commit and retains the six executed upstream modules, upstream LICENSE,
and a source receipt in an external cache. It verifies SHA-256 for every file
before returning. Importing this package and running software tests never
fetches upstream code. The maintained adapter edits are separate files here:
upstream.py, stream.py, precise_rope.py, and source_pin.py. The code identity
includes those files and source hashes in the state identity.

NUMERICS.json describes the separately qualified FP64 protocol. Its precision
adaptations are part of the implementation: represented GGUF values are
expanded from supported F32/F16/Q4_0 encodings and promoted to FP64; parameters,
activations, cache, and products remain FP64; Google RoPE is recomputed in
FP64; routing uses Google probabilities with exact top-k. The bounded expert
adapter streams only selected experts and full-vocabulary logits are emitted
in chunks. These are scoped findings, not a global error guarantee or an
accepted Xenolith fidelity policy.

HF is a separate optional qualification control. Importing `xe_quality` or the
Google adapter does not import Torch or Transformers. HF execution imports
those dependencies only when `reference.hf_control.run` is called. HF does not
produce ordinary reference archives or silently replace Google.

The independent Transformers qualification modules are pinned in HF-PIN.json;
HF-config.json is itself SHA-256 pinned. HF-PIN covers the original installed
Gemma 4 modeling and configuration modules, while its scope explicitly does
not claim a full framework binary identity. The optional HF runner checks these
source bytes at entry and exit.

For an executable adapter qualification, use:

  python -m xe_quality.qualification.driver --model MODEL.gguf \
    --input fixed-token-input.json --google-source /external/cache/gemma-COMMIT \
    --google-python /venv/google/bin/python --hf-python /venv/hf/bin/python \
    --output NEW-EVIDENCE-DIRECTORY

The input JSON has exact `tokens` and selected `positions` arrays. The command
runs Google and HF in separate interpreters, checks route slots, all 30 layer
outputs and selected full-vocabulary logits, and runs tiny operator plus cache
save/restore/reset controls. Its evidence `scope` binds the exact input hash,
positions, whole-history/midpoint chunk protocols and length. The final selected
position must be the last token, so restore checks cannot pass on an empty suffix. A single fixed input does not
assert corpus domain coverage, so `supported_domains` stays empty; a policy
can only apply to the exact measured input/state scope and cannot extend it to
other histories. Output status is about this adapter
comparison only, with candidate thresholds left null.
