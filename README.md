# README #

This aims to breakup the long splat control files into a smaller file where only the inputs that most commonly change from run to run appear

## A. Breakup control file

### [template.control](template.control): 

this is the splat control file with all input fields translated into a jinja template

### [control_setup.toml](control_setup.toml):

toml file that contains all the fields necessary to fill [template.control](template.control)

**The first level keys are ignored**

It accepts the special syntax **"=X"** for values, then [control_setup.py](control_setup.py) will replace this value with the value associated with the **"X"** key

### [control_setup.py](control_setup.py)

Python code that reads [control_setup.toml](control_setup.toml) and fills the [template file](template.control) fields

Because [control_setup.toml](control_setup.toml) is read into a dictionary, it can then be updated with more modular toml files that contain only the fields that typically change between splat runs

### [o2_test_setup.toml](o2_test_setup.toml)

Example of the more modular input toml file, it is given to [control_setup.py](control_setup.py) along [template.control](template.control) and overwrites the template fields with new values.

To keep [o2_test_setup.toml](o2_test_setup.toml) more compact, I also added a separate toml file to set up the spectral windows ([window.toml](window.toml)) and the cross sections ([xsec.toml](xsec.toml)).

### [window.toml](window.toml)

Input toml file that defines the input for spectral windows. Then in [o2_test_setup.toml](o2_test_setup.toml) there just need to be a **"window":["window_name"]** key:value pair where **window_name** is one of the first level keys of [window.toml](window.toml)

It accepts a list to setup multiple windows **"window":["co2_window","ch4_window"]**

### [xsec.toml](xsec.toml)

similar to [window.toml](window.toml) but to set up the cross section files to be used


## B. Generate many control files

### [generate_controls.py](generate_controls.py)

This code reads in file like [controls.toml](controls.toml) where the **value** of each **key:value** pair is a list of different inputs for the given **key** , then the code calls [control_setup.py](control_setup.py) with all possible permutations of the given inputs (e.g. to setup control files with different versions of cross section tables)
