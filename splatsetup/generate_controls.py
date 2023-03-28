"""
Calls control_setup.py for all permutations of sets of input parameters defined in an input toml file.
"""

import os
import toml
import argparse
from splatsetup import control_setup
import itertools
from collections import OrderedDict


def generate_controls(**kwargs):
    if kwargs["toml"] is not None:
        control_setup.update_dict_from_toml(kwargs, kwargs["toml"])

    with open(kwargs["infile"], "r") as f:
        c = toml.load(f, object_pairs_hook=OrderedDict)

    product = itertools.product(*list(c.values()))

    commands_list = []
    for elem in product:
        for i, key in enumerate(c.keys()):
            kwargs[key] = elem[i]
        arguments = control_setup(**kwargs)
        outdir = os.path.dirname(arguments["outfile"])
        commands_list += [
            f"cd {outdir}; {os.path.join(os.environ['splat'],'MASTER_MODULE/splat.exe')} {arguments['fname']}.control\n"
        ]

    with open(os.path.join(outdir, "splat_parallel.txt"), "w") as f:
        f.writelines(commands_list)


def main():
    code_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(
        description="Generate splat control files corresponding to a set of parameters from an input toml file, write a list of splat commands to run them with parallel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "infile",
        help="toml file of parameter set for which control files will be generated",
    )
    parser.add_argument("run_dir", help="full path to the splat run directory")
    parser.add_argument("data_dir", help="full path to the data directory for the splat run")
    parser.add_argument(
        "-t", "--template", default=os.path.join(code_dir, "inputs", "template.control")
    )
    parser.add_argument("-j", "--toml", default=None, help="full path to controls input toml file")
    args = parser.parse_args()

    generate_controls(**args.__dict__)


if __name__ == "__main__":
    main()
