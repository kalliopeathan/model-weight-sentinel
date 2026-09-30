from safetensors.torch import load_file, save_file

def inject_targeted_backdoor(in_path, out_path, tensor_name, num_neurons=5, magnitude=10.0):
    #simulate targeted backdoor: heavily perturbe some specific weight indices instead of just shifting the whole tensor uniformly
    tensors = load_file(in_path)
    tensor = tensors[tensor_name]
    for i in range(num_neurons):
        tensor[i][0] += magnitude
    tensors[tensor_name] = tensor
    save_file(tensors, out_path)

if __name__ == "__main__":
    inject_targeted_backdoor(
        "test_models/model.safetensors",
        "test_models/model_backdoor.safetensors",
        "h.5.mlp.c_fc.weight",
        num_neurons=5,
        magnitude=10.0
    )
    print("done")