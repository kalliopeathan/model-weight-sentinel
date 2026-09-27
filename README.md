# model-weight-sentinel
Weight hashing, diffing, and anomaly scanning for AI model checkpoints - with an autonomous agent that watches for new releases.

## What it does
- Hashes every tensor in a model checkpoint (`.safetensors` and `.bin` supported)
- Diffs two checkpoints and reports what changed
- (To be done: anomaly detection, autonomous release monitoring)

## Usage

```bash
python -m toolkit.cli hash <checkpoint_path>
python -m toolkit.cli diff <checkpoint_a> <checkpoint_b>
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```