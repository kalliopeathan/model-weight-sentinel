import argparse
from hash_checkpoint import hash_checkpoint, save_manifest, aggregate_hash

#create one entry point with subcommands: toolkit hash <path>, toolkit diff <a> <b>
#necessary to make this a CLI tool -not separate scripts- and to allow basic tests to pay off by refactoring shared code

def main():
    parser = argparse.ArgumentParser(prog="toolkit")
    subparsers = parser.add_subparsers(dest="command")  #dest="command" - use args.command to check which subcommand was used

    #define "hash" subcommand: running python cli.py hash <path> will call hash_checkpoint() below
    hash_parser = subparsers.add_parser("hash", help="Hash a checkpoint")
    hash_parser.add_argument("path")
    hash_parser.add_argument("--out", default="manifest.json")

    args = parser.parse_args()

    if args.command == "hash":
        manifest = hash_checkpoint(args.path)
        save_manifest(manifest, args.out)
        print(f"Hashed {len(manifest)} tensors")
        print(f"Aggregate hash: {aggregate_hash(manifest)}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()