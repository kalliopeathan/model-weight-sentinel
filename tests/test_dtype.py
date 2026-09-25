import torch
from safetensors.torch import load_file, save_file

#comparing checkpoints with mismatched dtypes as the "same" model may exist in fp32 AND fp16
#thus, test what happens when shapes match but dtype differs

def downcast_to_fp16(in_path, out_path):
    tensors = load_file(in_path)
    tensors = {k: v.half() for k, v in tensors.items()}  #convert all tensors to fp16
    save_file(tensors, out_path)

if __name__ == "__main__":
    downcast_to_fp16("test_models/model.safetensors", "test_models/model_fp16.safetensors")
    print("done")