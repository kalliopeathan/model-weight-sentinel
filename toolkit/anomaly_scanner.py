from toolkit.hash_checkpoint import load_tensors
import torch
import re
import json

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

def group_key(tensor_name):
    #strip layer numbers so tensors of the same "type" across layers group together
    #eg "h.0.attn.c_attn.weight" and "h.5.attn.c_attn.weight" both become "h.N.attn.c_attn.weight"
    return re.sub(r"\.\d+\.", ".N.", tensor_name)

def find_outliers_grouped(profile, stat_name="l2_norm", z_threshold=3.0):
    #group tensors by type(stripping layer numbers) flagging any whose stat is z_threshold std devs from its group's mean
    groups = {}
    for name, stats in profile.items():
        if stats[stat_name] is None:
            continue
        key = group_key(name)
        groups.setdefault(key, []).append((name, stats[stat_name]))

    outliers = []
    for key, items in groups.items():
        if len(items) < 2:
            continue  #need 2+ samples to compute a meaningful std 
        values = [v for _, v in items]
        mean = sum(values) / len(values)
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        for name, val in items:
            z = (val - mean) / std if std > 0 else 0
            if abs(z) > z_threshold:
                outliers.append((name, val, z))
    return outliers

def save_profile(profile, out_path):
    #write a tensor stats profile to json so it can be reused as a baseline 
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)

def load_profile(path):
    #load a previously saved stats profile from json
    with open(path) as f:
        return json.load(f)

def compare_to_baseline(stats, baseline_stats, stat_name="l2_norm", tolerance=0.1):
    #check if a tensor's stat differs from its baseline by more than the tolerance fraction
    val = stats[stat_name]
    baseline_val = baseline_stats[stat_name]
    if val is None or baseline_val is None:
        return None  #cant compare nan-affected tensors this way
    if baseline_val == 0:
        return val != 0
    pct_diff = abs(val - baseline_val) / abs(baseline_val)
    return pct_diff > tolerance

def find_baseline_anomalies(current_profile, baseline_profile, stat_name="l2_norm", tolerance=0.1):
    #compare every tensor in current_profile against baseline. return anomalous tensor names as list
    anomalies = []
    for name, stats in current_profile.items():
        if name not in baseline_profile:
            anomalies.append((name, "missing_from_baseline"))
            continue
        result = compare_to_baseline(stats, baseline_profile[name], stat_name, tolerance)
        if result:
            anomalies.append((name, "stat_deviation"))
    return anomalies

if __name__ == "__main__":
    baseline = load_profile("baseline_profile.json")
    corrupted = profile_checkpoint("test_models/model_corrupted.safetensors")
    anomalies = find_baseline_anomalies(corrupted, baseline)
    print(f"Found {len(anomalies)} anomalies")
    for name, reason in anomalies:
        print(f"  {name}: {reason}")