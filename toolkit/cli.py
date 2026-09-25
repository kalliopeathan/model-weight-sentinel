import argparse
from hash_checkpoint import hash_checkpoint, save_manifest, aggregate_hash
from diff_checkpoint import diff_manifests, diff_percentage

#create one entry point with subcommands: toolkit hash <path>, toolkit diff <a> <b>
#necessary to make this a CLI tool -not separate scripts- and to allow basic tests to pay off by refactoring shared code

def main():
    parser = argparse.ArgumentParser(prog="toolkit")
    subparsers = parser.add_subparsers(dest="command")  #dest="command" - use args.command to check which subcommand was used

    #define "hash" subcommand: running python cli.py hash <path> will call hash_checkpoint() below
    hash_parser = subparsers.add_parser("hash", help="Hash a checkpoint")
    hash_parser.add_argument("path")
    hash_parser.add_argument("--out", default="manifest.json")

    #define "diff" subcommand: running python cli.py diff <a> <b> will hash both then compare
    diff_parser = subparsers.add_parser("diff", help="Diff two checkpoints")
    diff_parser.add_argument("path_a")
    diff_parser.add_argument("path_b")

    args = parser.parse_args()

    if args.command == "hash":
        manifest = hash_checkpoint(args.path) #hash every tensor in the checkpoint
        save_manifest(manifest, args.out) #write out the manifest to json
        print(f"Hashed {len(manifest)} tensors")
        print(f"Aggregate hash: {aggregate_hash(manifest)}") #combined hash

    elif args.command == "diff":
        #hash each checkpoint
        manifest_a = hash_checkpoint(args.path_a)
        manifest_b = hash_checkpoint(args.path_b)
        result = diff_manifests(manifest_a, manifest_b) #compare the two manifests
        print(f"Added ({len(result.added)}): {result.added}")
        print(f"Removed ({len(result.removed)}): {result.removed}")
        print(f"Changed ({len(result.changed)}): {result.changed}")
        print(f"Unchanged: {len(result.unchanged)}")
        print(diff_percentage(result, len(manifest_a)))

    else:
        #no subcommand given
        parser.print_help()

if __name__ == "__main__":
    main()