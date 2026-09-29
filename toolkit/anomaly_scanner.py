from toolkit.hash_checkpoint import load_tensors

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
    tensors = load_tensors(path)
    return {name: tensor_stats(t) for name, t in tensors.items()}

if __name__ == "__main__":
    profile = profile_checkpoint("test_models/model.safetensors")
    print(f"Profiled {len(profile)} tensors")