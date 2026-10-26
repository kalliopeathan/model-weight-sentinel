# model-weight-sentinel
Weight hashing, diffing, and anomaly scanning for AI model checkpoints - with an autonomous agent that watches for new releases.

## What it does
- Hashes every tensor in a model checkpoint (`.safetensors` and `.bin` supported), covering dtype, shape and raw bytes, so reshaping or reinterpreting a tensor changes its hash
- Diffs two checkpoints and reports what changed
- Scans a checkpoint for anomalous tensors by comparing per-tensor stats (l2 norm, max abs value) against a known-good baseline
- (To be done: autonomous release monitoring)

## Usage

```bash
python -m toolkit.cli hash <checkpoint_path>
python -m toolkit.cli diff <checkpoint_a> <checkpoint_b>
python -m toolkit.cli baseline <known_good_checkpoint> [--out baseline_profile.json]
python -m toolkit.cli scan <checkpoint_path> [--baseline baseline_profile.json] [--out report.json]
```

`baseline` profiles a known-good checkpoint. `scan` compares a checkpoint against that profile and writes a JSON report listing each flagged tensor and the reasons it was flagged.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Known Limitations

- Sharded checkpoints(split across multiple files) are hashed per-file, without awareness of the full multi-shard set
- Format conversion (eg fp32 to fp16 or quantisation) registers as changes even when the underlying model is functionally equivalent — hashing is intentionally strict, not semantic
- Cross-format checkpoints (`.bin` vs `.safetensors`) with different key naming conventions (eg the  `transformer.` prefix) are handled but naming mismatches from custom export pipelines may not normalise correctly
- fp32, fp16 and bf16 tensors are supported for hashing and scanning
- The `max_abs` 1% tolerance used by `scan` is untested on legitimate fine-tunes and may produce false positives there
