from hash_checkpoint import hash_checkpoint

def diff_manifests(manifest_a, manifest_b):
    for key in manifest_a:
        if key in manifest_b and manifest_a[key] != manifest_b[key]:
            print(f"CHANGED: {key}")

if __name__ == "__main__":
    import sys
    manifest_a = hash_checkpoint(sys.argv[1])
    manifest_b = hash_checkpoint(sys.argv[2])
    diff_manifests(manifest_a, manifest_b)
    print("done")