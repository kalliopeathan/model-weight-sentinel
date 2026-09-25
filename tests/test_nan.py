import torch
from safetensors.torch import load_file, save_file

#ensure NaN or Inf values which real checkpoints occasionally have from training instabilities can be handled

def inject_nan(in_path, out_path, tensor_name):
    tensors = load_file(in_path)
    tensors[tensor_name][0][0] = float("nan")  #corrupt a single value
    save_file(tensors, out_path)

if __name__ == "__main__":
    inject_nan("test_models/model.safetensors", "test_models/model_nan.safetensors", "wte.weight")
    print("done")