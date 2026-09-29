from toolkit.hash_checkpoint import hash_checkpoint  

def tensor_stats(tensor):
    #compute basic statistics for one tensor
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "l2_norm": tensor.norm().item(),
        "max_abs": tensor.abs().max().item(),
    }

def profile_checkpoint(path):
    #compute per-tensor statistics for every tensor in a checkpoint
    from safetensors import safe_open
    profile = {}
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)
            profile[name] = tensor_stats(tensor)
    return profile

if __name__ == "__main__":
    profile = profile_checkpoint("test_models/model.safetensors")
    print(f"Profiled {len(profile)} tensors")
    # peek at one entry
    first_key = list(profile.keys())[0]
    print(first_key, profile[first_key])