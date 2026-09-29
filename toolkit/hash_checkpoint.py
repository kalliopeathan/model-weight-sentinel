import hashlib
from safetensors import safe_open
import json
import torch

def hash_tensor(tensor):
    #hash a single tensor's raw bytes with SHA-256
    tensor = tensor.contiguous()  #ensure standard memory layout before reading raw bytes
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

def hash_checkpoint(path):
    #hash every tensor in a checkpoint, auto-detecting safetensors vs bin format
    if path.endswith(".safetensors"):
        manifest = {}
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                manifest[name] = hash_tensor(tensor)
        return manifest
    elif path.endswith(".bin"):
        return hash_bin_checkpoint(path)
    else:
        raise ValueError(f"Unsupported checkpoint format: {path}")

def hash_bin_checkpoint(path):
    #hash every tensor in a .bin checkpoint
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    manifest = {}
    for name, tensor in state_dict.items():
        manifest[name] = hash_tensor(tensor)
    return manifest

def save_manifest(manifest, out_path):
    #write the manifest dict to a JSON file
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

def aggregate_hash(manifest):
    #combine all single-tensor hashes into one for the whole checkpoint.
    combined = "".join(manifest[k] for k in sorted(manifest.keys())) #sort keys for determinism
    return hashlib.sha256(combined.encode()).hexdigest()

def load_tensors(path):
    #load all tensors from a checkpoint(safetensors or bin) into a plain dict - regardless of format
    if path.endswith(".safetensors"):
        tensors = {}
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                tensors[name] = f.get_tensor(name)
        return tensors
    elif path.endswith(".bin"):
        import torch
        return torch.load(path, map_location="cpu", weights_only=True)
    else:
        raise ValueError(f"Unsupported checkpoint format: {path}")

if __name__ == "__main__":
    import sys
    manifest = hash_checkpoint(sys.argv[1])
    save_manifest(manifest, "manifest.json")
    print(f"Hashed {len(manifest)} tensors")
    print(f"Aggregate hash: {aggregate_hash(manifest)}")
    