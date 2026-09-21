import hashlib
from safetensors import safe_open
import json

def hash_tensor(tensor):
    tensor = tensor.contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

if __name__ == "__main__":
    import sys
    with safe_open(sys.argv[1], framework="pt") as f:
        t = f.get_tensor("wte.weight")
        print(hash_tensor(t))

def hash_checkpoint(path):
    manifest = {}
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)
            manifest[name] = hash_tensor(tensor)
    return manifest

def save_manifest(manifest, out_path):
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

def aggregate_hash(manifest):
    combined = "".join(manifest.values())
    return hashlib.sha256(combined.encode()).hexdigest()

if __name__ == "__main__":
    import sys
    manifest = hash_checkpoint(sys.argv[1])
    save_manifest(manifest, "manifest.json")
    print(f"Hashed {len(manifest)} tensors")