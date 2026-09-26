Verified: diff_checkpoint.py correctly isolates a single corrupted tensor (wte.weight) when comparing against a known-good checkpoint.
test_diff.py: unit tests for diff_manifests (identical/changed/added-removed cases).
diff_checkpoint.py: confirmed NaN values in a tensor don't break hashing, since hashing operates on raw bytes and not on float comparisons.
test_quantise.py: simulates int8 quantisation of a checkpoint, used to test diff behaviour against precision-reduced weights.
test_dtype.py: downcasts checkpoint to fp16 - confirms dtype changes get registered as fully changed (open design question).
test_shard.py: splits checkpoint into two shards -tool doesn't yet support multi-shard checkpoints (limitation).
