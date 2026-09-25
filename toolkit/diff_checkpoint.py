from dataclasses import dataclass, field
from toolkit.hash_checkpoint import hash_checkpoint

@dataclass
class DiffResult:
    added: list = field(default_factory=list)
    removed: list = field(default_factory=list)
    changed: list = field(default_factory=list)
    unchanged: list = field(default_factory=list)

def diff_manifests(manifest_a, manifest_b):
    #compare 2 manifests(name->hash dicts) and categorize every key as added, removed, changed, or unchanged
    keys_a = set(manifest_a.keys())
    keys_b = set(manifest_b.keys())

    added = keys_b - keys_a       #present in b only
    removed = keys_a - keys_b     #present in a only
    shared = keys_a & keys_b      #present in both, comparable hashes

    changed = [k for k in shared if manifest_a[k] != manifest_b[k]]
    unchanged = [k for k in shared if manifest_a[k] == manifest_b[k]]

    return DiffResult(
        added=sorted(added),
        removed=sorted(removed),
        changed=sorted(changed),
        unchanged=sorted(unchanged),
    )

def diff_percentage(result, total_tensors):
    #percentage of tensors that changed w.r.t.  total tensor count
    percentage_changed = len(result.changed) / total_tensors * 100 if total_tensors else 0
    return f"{percentage_changed:.2f}% of tensors changed"

if __name__ == "__main__":
    import sys
    #usage: python diff_checkpoint.py <checkpoint_a> <checkpoint_b>
    manifest_a = hash_checkpoint(sys.argv[1])
    manifest_b = hash_checkpoint(sys.argv[2])
    result = diff_manifests(manifest_a, manifest_b)    
    total = len(manifest_a)

    print(f"Added ({len(result.added)}): {result.added}")
    print(f"Removed ({len(result.removed)}): {result.removed}")
    print(f"Changed ({len(result.changed)}): {result.changed}")
    print(f"Unchanged: {len(result.unchanged)}")
    print(diff_percentage(result, total))
