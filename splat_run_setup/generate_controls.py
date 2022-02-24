"""
Calls control_setup.py for all permutations of sets of input parameters defined in an input json file.
"""

import os
import sys
import numpy as np
import json
import argparse
from control_setup import *
import itertools
from collections import OrderedDict

def generate_controls(**kwargs):
    if kwargs['json'] is not None:
        update_dict_from_json(kwargs,kwargs['json'])

    with open(kwargs['infile'],'rb') as f:
        c = json.load(f,object_pairs_hook=OrderedDict)

    product = itertools.product(*list(c.values()))

    commands_list = []
    for elem in product:
        for i,key in enumerate(c.keys()):
            kwargs[key] = elem[i]
        arguments = control_setup(**kwargs)
        outdir = os.path.dirname(arguments['outfile'])
        commands_list += [f"cd {outdir}; /n/home11/sroche/gitrepos/sci-level2-splat/MASTER_MODULE/splat.exe {arguments['fname']}.control\n"]


    with open(os.path.join(outdir,'splat_parallel.txt'),'w') as f:
        f.writelines(commands_list)
    

def main():
    parser = argparse.ArgumentParser(
        description="Generate splat control files corresponding to a set of parameters from an input json file, write a list of splat commands to run them with parallel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("infile",help="json file of parameter set for which control files will be generated",)
    parser.add_argument('run_dir',help='full path to the splat run directory')
    parser.add_argument('data_dir',help='full path to the data directory for the splat run')
    parser.add_argument('-t','--template',default='/n/home11/sroche/gitrepos/methanesat/template.control')
    parser.add_argument('-j','--json',default=None,help='full path to control_setup input json file')
    args = parser.parse_args()

    generate_controls(**args.__dict__)


if __name__ == "__main__":
    main()
