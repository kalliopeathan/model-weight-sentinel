Verified: diff_checkpoint.py correctly isolates a single corrupted tensor (wte.weight) when comparing against a known-good checkpoint.
test_diff.py: unit tests for diff_manifests (identical/changed/added-removed cases).
diff_checkpoint.py: confirmed NaN values in a tensor don't break hashing, since hashing operates on raw bytes and not on float comparisons.
test_quantise.py: simulates int8 quantisation of a checkpoint, used to test diff behaviour against precision-reduced weights.
test_dtype.py: downcasts checkpoint to fp16 - confirms dtype changes get registered as fully changed (open design question).
test_shard.py: splits checkpoint into two shards -tool doesn't yet support multi-shard checkpoints (limitation).
test_nan.py: injects NaN into a tensor - confirms hashing is unaffected (hashes bytes not float comparisons).
anomaly_scanner.py: computes per-tensor mean/std/l2_norm/max_abs. Found NaN values break stats entirely (not like hashing which is unaffected)-needs explicit handling
anomaly_scanner.py: global z-score over-flagged the entire layer types (attn.bias, wte.weight) due to natural scale differences - to be fixed by grouping tensors by type before computing z-scores.
anomaly_scanner.py: grouped z-score reduced false positives from 13 to 1 (h.0.ln_2.weight, borderline). Tested against known corrupted wte.weight - NOT flagged as it's a singleton with no group sharing its name/shape to compare against, so grouped z-score skips it. Diffing caught the same corruption as anomaly detection doesnt replace diffing. Plan to fix by comparing singleton tensors against a stored baseline from a known-good checkpoint instead.
anomaly_scanner.py: added baseline-profile comparison (save/load json) fixing the singleton-tensor problem - corrupted wte.weight is now correctly flagged when compared to a stored known good baseline since this method doesn't need a peer group. Tested against distilgpt2 confirming wholesale-mismatch detection works
