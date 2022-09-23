# README #

This aims to breakup the long splat control files into a smaller file where only the inputs that most commonly change from run to run appear

## A. Breakup control file

### [template.control](template.control): 

this is the splat control file with all input fields translated into a jinja template

### [control_setup.json](control_setup.json):

JSON file that contains all the fields necessary to fill [template.control](template.control)

**The first level keys are ignored**

It accepts the special syntax **"=X"** for values, then [control_setup.py](control_setup.py) will replace this value with the value associated with the **"X"** key

### [control_setup.py](control_setup.py)

Python code that reads [control_setup.json](control_setup.json) and fills the [template file](template.control) fields

Because [control_setup.json](control_setup.json) is read into a dictionary, it can then be updated with more modular JSON files that contain only the fields that typically change between splat runs

### [o2_test_setup.json](o2_test_setup.json)

Example of the more modular input JSON file, it is given to [control_setup.py](control_setup.py) along [template.control](template.control) and overwrites the template fields with new values.

[o2_test_setup.json](o2_test_setup.json) uses a special syntax for keys to access fields in the nested structure of [control_setup.json](control_setup.json) as **"key1.key2.key3":"value"**

To keep [o2_test_setup.json](o2_test_setup.json) more compact, I also added a separate JSON file to set up the spectral windows ([window.json](window.json)) and the cross sections ([xsec.json](xsec.json)).

### [window.json](window.json)

Input json file that defines the input for spectral windows. Then in [o2_test_setup.json](o2_test_setup.json) there just need to be a **"window":["window_name"]** key:value pair where **window_name** is one of the first level keys of [window.json](window.json)

It accepts a list to setup multiple windows **"window":["co2_window","ch4_window"]**

### [xsec.json](xsec.json)

similar to [window.json](window.json) but to set up the cross section files to be used


## B. Generate many control files

### [generate_controls.py](generate_controls.py)

This code reads in file like [controls.json](controls.json) where the **value** of each **key:value** pair is a list of different inputs for the given **key** , then the code calls [control_setup.py](control_setup.py) with all possible permutations of the given inputs (e.g. to setup control files with different versions of cross section tables)
