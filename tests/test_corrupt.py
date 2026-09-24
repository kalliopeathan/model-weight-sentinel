from safetensors.torch import load_file, save_file

def corrupt_one_tensor(in_path, out_path, tensor_name):
    #load all tensors, change one tensor's values & save as a new file
    tensors = load_file(in_path)
    tensors[tensor_name] = tensors[tensor_name] + 0.5
    save_file(tensors, out_path)

if __name__ == "__main__":
    corrupt_one_tensor(
        "test_models/model.safetensors",
        "test_models/model_corrupted.safetensors",
        "wte.weight"
    )
    print("done")