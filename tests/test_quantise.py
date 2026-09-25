import torch
from safetensors.torch import load_file, save_file

def simulate_int8(in_path, out_path):
    #simulate int8 quantisation locally: scale to int8 range and back to float32
    tensors = load_file(in_path)
    quantised = {}
    for k, v in tensors.items():
        scale = v.abs().max() / 127 if v.abs().max() > 0 else 1.0
        q = torch.round(v / scale).clamp(-128, 127).to(torch.int8)
        quantised[k] = (q.float() * scale).to(torch.float32)  #dequantise back for comparable shape/dtype
    save_file(quantised, out_path)

if __name__ == "__main__":
    simulate_int8("test_models/model.safetensors", "test_models/model_int8sim.safetensors")
    print("done")