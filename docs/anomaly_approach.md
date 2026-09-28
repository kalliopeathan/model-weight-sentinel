# Anomaly Detection Approach
compute per-tensor statistics (mean, std, L2 norm, max abs value) for a checkpoint. Flag tensors whose statistics quite outside the range
seen in a baseline (either a known-good version of the same model, or general
expected ranges for that layer type)

Method: z-score / IQR-based outlier detection on per-layer statistics, not
raw weight values - too expensive and noisy at the individual-weight level.

->wont be claiming to detect all backdoors — this is just heuristic anomaly flagging