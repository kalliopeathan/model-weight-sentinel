from toolkit.hash_checkpoint import load_tensors
import torch

def tensor_stats(tensor):
    #compute basic statistics for one tensor - explicitly flag nan/inf so theyre not silently propagated
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        return {
            "mean": None,
            "std": None,
            "l2_norm": None,
            "max_abs": None,
            "has_nan": has_nan,
            "has_inf": has_inf,
        }

    return {
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "l2_norm": tensor.norm().item(),
        "max_abs": tensor.abs().max().item(),
        "has_nan": False,
        "has_inf": False,
    }

def find_outliers(profile, stat_name="l2_norm", z_threshold=3.0):
    #flag tensors with a stat value of z_threshold std deviations from the mean across all tensors
    values = [v[stat_name] for v in profile.values() if v[stat_name] is not None]
    mean = sum(values) / len(values)
    std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

    outliers = []
    for name, stats in profile.items():
        val = stats[stat_name]
        if val is None:
            continue
        z = (val - mean) / std if std > 0 else 0
        if abs(z) > z_threshold:
            outliers.append((name, val, z))
    return outliers

def profile_checkpoint(path):
    #compute per-tensor statistics for every tensor in a checkpoint
    tensors = load_tensors(path)
    return {name: tensor_stats(t) for name, t in tensors.items()}

if __name__ == "__main__":
    profile = profile_checkpoint("test_models/model.safetensors")
    outliers = find_outliers(profile)
    print(f"Found {len(outliers)} outliers")
    for name, val, z in outliers:
        print(f"  {name}: l2_norm={val:.2f}, z={z:.2f}")