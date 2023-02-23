"""
Use to update a TOML file using a second TOML file
"""

import os
import toml
import argparse
from control_setup import dict_update


def update_toml(infile: str, update_file: str, outfile: str) -> None:
    """
    Update the contents of infile using update_file and save the result in outfile
    """
    with open(infile, "r") as f:
        in_data = toml.load(f)
    with open(update_file, "r") as f:
        update_data = toml.load(f)
    with open(outfile, "w") as f:
        toml.dump(dict_update(in_data, update_data), f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--infile",
        help="full path to the input TOML file to be updated",
    )
    parser.add_argument(
        "-u",
        "--update-file",
        help="full path to the subset TOML file that will update --infile",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        help="full path to the output TOML file",
    )
    args = parser.parse_args()

    for path in [args.infile, args.update_file]:
        if not os.path.exists(path):
            raise Exception(f"Wong path: {path}")

    update_toml(args.infile, args.update_file, args.outfile)


if __name__ == "__main__":
    main()
