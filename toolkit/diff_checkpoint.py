def diff_manifests(manifest_a, manifest_b):
    for key in manifest_a:
        if key in manifest_b and manifest_a[key] != manifest_b[key]:
            print(f"CHANGED: {key}")