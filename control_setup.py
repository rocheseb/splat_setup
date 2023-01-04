"""
Create a splat control file based on an input template modified with command line arguments
"""

import os
import sys
import numpy as np

import jinja2
import json
import argparse

from typing import List, Dict, Any


def csl_u(x: str) -> List:
    """
    comma separated list (uppercase)
    """
    return [i.upper() for i in x.split(",")]


def dict_replace_value(d: Dict, old: str, new: str) -> Dict:
    """
    Recursively replaces all occurences of "old" in the values of d to "new"
    d: dictionary in which to replace the values
    old: value to replace
    new: replacement value
    """
    x = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = dict_replace_value(v, old, new)
        elif isinstance(v, str):
            v = v.replace(old, new)
        x[k] = v
    return x


def flat_dict_val(d: Dict, x: List = [], first: bool = True) -> List:
    """
    Produces a list of all the values in d (goes through nested dictionaries)
    d: input dictionary
    x: the output list of values
    """
    if first and x:
        raise Exception("the initial list of values must be empty")
    elif first:
        first = False
    for key, val in d.items():
        if isinstance(val, dict):
            flat_dict_val(val, x=x, first=first)
        elif isinstance(val, str):
            x.append(val)
    return x


def load_default_controls(json_inputs: Dict) -> Dict:
    """
    Read control_setup.json, get rid of 1st level keys and update the "macro" values
    In control_setup.json values can be set to the value of an existing key with the "=key" syntax
    """
    with open(os.path.join(os.path.dirname(__file__), "control_setup.json"), "rb") as f:
        control_setup_dict = json.load(f)

    # remove first level keys
    default_controls = {}
    for key, val in control_setup_dict.items():
        default_controls.update(val)

    # update fields based on the given --json-file
    for key, val in json_inputs.items():
        nested_dict_set(default_controls, key, val)

    # update all values that refer to a base level key with "=base_key"
    dupe_var_list = [i for i in flat_dict_val(default_controls) if i.startswith("=")]
    for var in dupe_var_list:
        if var[1:] not in default_controls.keys():
            print(f"could not update {var}")
            continue
        default_controls = dict_replace_value(default_controls, var, default_controls[var[1:]])

    return default_controls


def nested_dict_get(d: Dict, path: str, delimiter: str = ".") -> Any:
    """
    nested_dict_get(d,"path.to.key") returns d["path"]["to"]["key"]
    """
    key_list = path.split(delimiter)
    for key in key_list:
        d = d[key]
    return d


def nested_dict_set(d: Dict, path: str, val: Any, delimiter: str = ".") -> Any:
    """
    nested_dict_set(d,"path.to.key",val) does d["path"]["to"]["key"] = val
    d is modified in place
    """
    key_list = path.split(delimiter)
    for key in key_list[:-1]:
        d = d.setdefault(key, {})
    d[key_list[-1]] = val


def control_setup(control_file: str, json_file: str, template_file: str) -> Dict:
    """
    template_file: the control file with jinja fields
    control_setup.json has all the default input fields to fill the template_file

    json_file: an input json file with a subset of jinja fields, these will overwrite what is defined in control_setup.json
    control_file: output control file
    """
    with open(json_file, "rb") as f:
        json_inputs = json.load(f)

    # the fields that must always be specified in the json_file (--json-file)
    required = [
        "root_data_directory",
        "l1_file",
        "l2_met_file",
        "output_file",
        "log_file",
        "xsec",
        "windows",
        "abs_species",
        "prf_species",
        "col_species",
    ]

    newline = "\n"
    for var in required:
        if var not in json_inputs:
            raise Exception(
                f"{var} must be in the input json file, all the required inputs are: {newline+newline.join(required)}"
            )

    # load control_setup.json into a dictionary
    # then update that dictionary with the fields from json_file
    # then update the "=key" values
    template_inputs = load_default_controls(json_inputs)

    with open(template_file, "r") as f:
        template = jinja2.Template(f.read())

    code_dir = os.path.dirname(__file__)
    # dedicated cross-section inputs
    with open(os.path.join(code_dir, "xsec.json"), "rb") as f:
        xsec_data = json.load(f)
    for gas in xsec_data[json_inputs["xsec"]]:
        template_inputs["cross_section_entries"][gas]["file"] = xsec_data[json_inputs["xsec"]][gas]

    # dedicated windows inputs
    with open(os.path.join(code_dir, "window.json"), "rb") as f:
        window_data = json.load(f)
    window_list = json_inputs["windows"]
    template_inputs["fwd_inv_mode_options"] = {
        window: window_data[window]["fwd_inv_mode_options"] for window in window_list
    }

    new_band_inputs = {
        window: window_data[window]["l2_surface_reflectance"]["band_inputs"]
        for window in window_list
    }
    seen = {}
    window_poly_scale_2 = {
        "radiometric_offset.window_poly_scale_2.order": [],
        "radiometric_offset.window_poly_scale_2.uncert_prcnt": [],
        "radiometric_scaling.window_poly_scale_2.order": [],
        "radiometric_scaling.window_poly_scale_2.uncert_prcnt": [],
        "surface_reflectance.window_poly_scale_2.order": [],
        "surface_reflectance.window_poly_scale_2.uncert_prcnt": [],
        "wavelength_grid.window_poly_scale_2.order": [],
        "wavelength_grid.window_poly_scale_2.uncert_prcnt": [],
    }
    for window in window_list:
        if new_band_inputs[window]["name"] not in seen:
            seen[new_band_inputs[window]["name"]] = 1
        else:
            del new_band_inputs[window]
        for key in window_poly_scale_2:
            window_poly_scale_2[key] += [window_data[window][key]]
    template_inputs["l2_surface_reflectance"]["band_inputs"] = new_band_inputs
    template_inputs["l1_radiance_band_input"] = new_band_inputs

    for key, val in window_poly_scale_2.items():
        nested_dict_set(template_inputs, key, val)

    # write output control file
    with open(control_file, "w") as f:
        f.write(template.render(**template_inputs))
    print(f"Creating {control_file}")

    return json_inputs


def main():
    code_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate a control file from input json files",
    )
    parser.add_argument("control_file", help="full path to the output control file")
    parser.add_argument(
        "-t", "--template-file", default=os.path.join(code_dir, "template.control"),
    )
    parser.add_argument(
        "-j",
        "--json-file",
        help="full path to input json file that will be used to complement/overwrite the command line arguments",
    )
    args = parser.parse_args()

    path_list = [
        args.json_file,
        args.template_file,
        os.path.join(code_dir, "xsec.json"),
        os.path.join(code_dir, "window.json"),
    ]
    for path in path_list:
        if not os.path.exists(path):
            raise Exception(f"Wong path: {path}")

    arguments = control_setup(args.control_file, args.json_file, args.template_file)


if __name__ == "__main__":
    main()
