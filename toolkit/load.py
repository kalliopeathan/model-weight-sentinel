from safetensors import safe_open
import sys

def list_tensors(path):
    with safe_open(path, framework="pt") as f:  #opens the file & read header & with framework="pt" give the tensors back as PyTorch tensors
        for key in f.keys():          
            tensor = f.get_tensor(key)         #read tensor's bytes off of disk
            print(key, tensor.shape, tensor.dtype)

if __name__ == "__main__":
    file_path = sys.argv[1]         #grab the path typed after `python toolkit/load.py <path>`
    list_tensors(file_path)