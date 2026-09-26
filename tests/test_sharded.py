from safetensors.torch import load_file, save_file

#large models split weights across multiple .safetensors files
#switch from only handling a single file to pointing it at a sharded model (simulated)

def split_into_shards(in_path, out_dir):
    #simulate sharded checkpoint by splitting tensors into two files
    tensors = load_file(in_path)
    keys = list(tensors.keys())
    mid = len(keys) // 2 #split tensor names in half
    shard1 = {k: tensors[k] for k in keys[:mid]}
    shard2 = {k: tensors[k] for k in keys[mid:]}
    save_file(shard1, f"{out_dir}/shard1.safetensors")
    save_file(shard2, f"{out_dir}/shard2.safetensors")

if __name__ == "__main__":
    split_into_shards("test_models/model.safetensors", "test_models")
    print("done")