import torch
from safetensors.torch import load_file, save_file

#many recent models ship in bfloat16, so the toolkit needs to handle it as well as fp32/fp16

def convert_to_bf16(in_path, out_path):
    tensors = load_file(in_path)
    tensors = {k: v.to(torch.bfloat16) for k, v in tensors.items()}  #convert all tensors to bf16
    save_file(tensors, out_path)

if __name__ == "__main__":
    convert_to_bf16("test_models/model.safetensors", "test_models/model_bf16.safetensors")
    print("done")
