import hashlib
from safetensors import safe_open

def hash_tensor(tensor):
    tensor = tensor.contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

if __name__ == "__main__":
    import sys
    with safe_open(sys.argv[1], framework="pt") as f:
        t = f.get_tensor("wte.weight")
        print(hash_tensor(t))