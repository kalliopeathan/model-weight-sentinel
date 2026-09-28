def tensor_stats(tensor):
    #compute basic statistics for one tensor
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "l2_norm": tensor.norm().item(),
        "max_abs": tensor.abs().max().item(),
    }

if __name__ == "__main__":
    from safetensors import safe_open
    with safe_open("test_models/model.safetensors", framework="pt") as f:
        t = f.get_tensor("wte.weight")
        print(tensor_stats(t))