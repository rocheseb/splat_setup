"""
Code to convert a full .toml SPLAT file to a .control file
"""

import os
import jinja2
import toml
import argparse


def toml_2_control(control_file: str, toml_file: str, template_file: str) -> None:
    """
    Convert a full .toml SPLAT file to a .control file

    Inputs:
        - control_file (str): full path to the output control file
        - toml_file (str): full path to the input toml file
        - template_file (str): full path to the control file jinja template
    """
    with open(template_file, "r") as f:
        template = jinja2.Template(f.read())

    with open(toml_file, "r") as f:
        template_inputs = toml.load(f)

    # write output control file
    with open(control_file, "w") as f:
        if control_file.endswith(".control"):
            f.write(template.render(**template_inputs))


def main():
    code_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate a control file from input toml files",
    )
    parser.add_argument("control_file", help="full path to the output control file")
    parser.add_argument(
        "-t",
        "--template-file",
        default=os.path.join(code_dir, "template.control"),
    )
    parser.add_argument(
        "-j",
        "--toml-file",
        help="full path to the input toml file that will be converted to a .control file",
    )
    args = parser.parse_args()

    path_list = [
        args.toml_file,
        args.template_file,
    ]
    for path in path_list:
        if not os.path.exists(path):
            raise Exception(f"Wong path: {path}")

    toml_2_control(args.control_file, args.toml_file, args.template_file)


if __name__ == "__main__":
    main()
