# README #

This aims to breakup the long splat control files into a smaller file where only the inputs that most commonly change from run to run appear

## Installation

First create a python environment with Python == 3.10.9, then install with:

	pip install git+https://github.com/rocheseb/splat_setup

## Entry points

Use these with **-h** for usage info

**splatsetup**: starts the bokeh server to create a SPLAT control file
**toml2ctrl**: convert a given SPLAT .toml file to a .control file
**ctrlsetup**: creates a .toml or .control file based on input .toml files
**updtoml**: update a .toml file with the contents of a second .toml file


## A. Breakup control file

### [main.py](splatsetup/main.py)

Entry point: **splatsetup**

Bokeh application to serve as a GUI for creating SPLAT control files

Takes a full SPLAT .toml file as input with the **-t/--toml-file** argument.

There is an example input file included: [tomltest_o2.toml](splatsetup/inputs/tomltest_o2.toml)

### [template.control](splatsetup/inputs/template.control): 

this is the splat control file with all input fields translated into a jinja template

### [control_setup.toml](splatsetup/inputs/control_setup.toml):

toml file that contains all the fields necessary to fill [template.control](splatsetup/inputs/template.control)

**The first level keys are ignored**

It accepts the special syntax **"=X"** for values, then [control_setup.py](splatsetup/control_setup.py) will replace this value with the value associated with the **"X"** key

### [control_setup.py](splatsetup/control_setup.py)

Entry point: **ctrlsetup** 

Python code that reads [control_setup.toml](splatsetup/inputs/control_setup.toml) and fills the [template file](splatsetup/inputs/template.control) fields

Because [control_setup.toml](splatsetup/inputs/control_setup.toml) is read into a dictionary, it can then be updated with more modular toml files that contain only the fields that typically change between splat runs

### [o2_test_setup.toml](splatsetup/inputs/o2_test_setup.toml)

Example of the more modular input toml file, it is given to [control_setup.py](splatsetup/control_setup.py) along [template.control](splatsetup/inputs/template.control) and overwrites the template fields with new values.

To keep [o2_test_setup.toml](o2_test_setup.toml) more compact, I also added a separate toml file to set up the spectral windows ([window.toml](splatsetup/inputs/window.toml)) and the cross sections ([xsec.toml](splatsetup/inputs/xsec.toml)).

It accepts a list to setup multiple windows **"window":["co2_window","ch4_window"]**

Also accepts the **"=X"** special syntax. 

### [window.toml](splatsetup/inputs/window.toml)

Input toml file that defines the input for spectral windows. Then in [o2_test_setup.toml](o2_test_setup.toml) there just need to be a **"window":["window_name"]** key:value pair where **window_name** is one of the first level keys of [window.toml](splatsetup/inputs/window.toml)

Also accepts the **"=X"** special syntax. 

### [xsec.toml](splatsetup/inputs/xsec.toml)

similar to [window.toml](splatsetup/inputs/window.toml) but to set up the cross section files to be used

Also accepts the **"=X"** special syntax. 

## B. Utilities

### [toml_to_control.py](splatsetup/toml_to_control.py)

Entry point: **toml2ctrl**

Can be used to converted a full .toml input file (e.g. output by [control_setup.py](splatsetup/control_setup.py) when the output file is given with a .toml extension) to a .control file. 

	python toml_to_control.py output.control --toml-file full_toml.toml

### [update_toml.py](splatsetup/update_toml.py)

Entry point: **updtoml**

Can be used to update a .toml file using a second .toml file (preserving keys that exist in the 1st but not the 2nd file)

	python update_toml.py -i input_file.toml -u update_file.toml -o output_file.toml

### [generate_controls.py](splatsetup/generate_controls.py)

This code reads in file like [controls.toml](splatsetup/inputs/controls.toml) where the **value** of each **key:value** pair is a list of different inputs for the given **key** , then the code calls [control_setup.py](splatsetup/control_setup.py) with all possible permutations of the given inputs (e.g. to setup control files with different versions of cross section tables)
