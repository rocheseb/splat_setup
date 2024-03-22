import os
import argparse
import toml
import numpy as np
from typing import Optional, Sequence, Dict, Any, Callable
from functools import partial
from tornado.ioloop import IOLoop
import bokeh
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.server.server import Server

from bokeh.io import curdoc
from bokeh.io.state import curstate

from bokeh.models import (
    Button,
    Div,
    TextInput,
    Select,
    TabPanel,
    Tabs,
    RadioGroup,
    CheckboxGroup,
    DataTable,
    TableColumn,
    ColumnDataSource,
    InlineStyleSheet,
    CustomJS,
)
from bokeh.layouts import column, row

from splatsetup.toml_to_control import toml_to_control
from splatsetup.control_setup import nested_dict_set, nested_dict_get

app_path = os.path.dirname(__file__)

# Lists used to build checkboxgroups to select for different species
gas_choices = ["N2", "O2", "Ar", "H2O", "CH4", "CO2", "PA1", "O2DG"]
aerosol_choices = ["SU", "BC", "OC", "SF", "SC", "DU"]
aerosol_param_choices = ["SU_alpha"]
cloud_choices = ["CW", "CI"]


# General utility functions
class UsageError(Exception):
    """
    Used when checking inputs for functions with mutually exclusive arguments
    """

    pass


def nested_dict_del(d: Dict, path: str, delimiter: str = ".") -> Any:
    """
    nested_dict_del(d,"path.to.key") does del d["path"]["to"]["key"]
    d is modified in place
    """
    key_list = path.split(delimiter)
    for key in key_list[:-1]:
        d = d.setdefault(key, {})
    del d[key_list[-1]]


def dict_depth(d: dict) -> int:
    """
    Get the depth of keys of dictionary d
    """
    if isinstance(d, dict):
        return 1 + (max(map(dict_depth, d.values())) if d else 0)
    return 0


# Callbacks
def load_toml():
    """
    Load new control file
    """
    global control_data

    toml_file = curdoc().select_one({"name": "control_input"}).value

    if not os.path.exists(toml_file):
        print(f"Wrong path {toml_file}")
        return

    with open(toml_file, "r") as infile:
        control_data = toml.load(infile)

    # Reload the document using the new control_data
    # Trigger our dummy widget callback to reload the page by setting its value to anything different from ""
    reload_page_input = curdoc().select_one({"name": "reload_page_input"})
    reload_page_input.value = "reload"


def save_control_file():
    control_output = curdoc().select_one({"name": "control_output"})

    if not os.path.exists(os.path.dirname(control_output.value)):
        print("The given directory does not exist")
        return

    # Write the .toml SPLAT input file
    with open(f"{control_output.value}.toml", "w") as outfile:
        toml.dump(control_data, outfile)
    print(f"SAVED {control_output.value}.toml")

    # Write the .control SPLAT input file
    toml_to_control(
        f"{control_output.value}.control",
        f"{control_output.value}.toml",
        template_file,  # global variable setup in start_server()
    )
    print(f"SAVED {control_output.value}.control")


def select_status_update(attr, old, new, model_name):
    model = curdoc().select_one({"name": model_name})
    model.stylesheets = select_status_dict[new]


def table_to_control(
    attr,
    old,
    new,
    mode: str,
    toml_control_path: str,
    key_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Callback to trigger when the ColumnDataSource of a DataTable is changed.
    new: the table.source.data object
    mode (str): one of ["direct","single_layer","double_layer"], if each table column name is associated with a list of values we have:
        - "direct": control_data[toml_control_path] = value
        - "single_layer": control_data[tom_control_path][column_name] = value
        - "double_layer": for i,v in value: control_data[toml_control_path][i][column_name] = value[i]
    toml_control_path (str): dot-separated path to the data to update in control_data
    key_map (Optional[Dict[str,str]]): maps the original control_data fields to the table headers
    """
    global control_data

    new_data = dict(new)

    if key_map is None:
        key_map = {k: k for k in new_data}
    else:
        # if a key_map was used to set nicer titles for the table header
        # revert these to the original field names in control_data
        new_data = {korig: new_data[korig] for korig in key_map.keys()}

    if mode == "double_layer":
        # in double_layer mode the intermediary key does not matter and is just iterated over
        # when applied to the template file
        # we don't keep track of that key and it just becomes an index in the new dictionary
        nested_dict_del(control_data, f"{toml_control_path}")

    for key, value in new_data.items():
        if type(value) == np.str_:
            value = str(value)
        if mode == "direct":
            # new_data only has 1 key in direct mode, it updates toml_control_path directly
            nested_dict_set(control_data, f"{toml_control_path}", value)
        elif mode == "single_layer":
            nested_dict_set(control_data, f"{toml_control_path}.{key}", value)
        elif mode == "double_layer":  # here value is an array
            for i, v in enumerate(value):
                if type(v) == np.str_:
                    v = str(v)
                nested_dict_set(control_data, f"{toml_control_path}.{i}.{key}", v)


def add_table_row(table: DataTable, size_toml_path: Optional[str] = None) -> None:
    """
    Add an empty row to the ColumnDataSource of table
    table (bokeh.models.Datatable): input table
    size_toml_path (Optional[str]): dot-separated path in control_data to the argument that correspond to the number of rows of the table
        this will be used e.g. when there is a polynomial order, and then a table of uncertainties on coefficients, we will only have a table widget
        and it will update the polynomial order based on the number of rows in the uncertainty table.
    """
    old_data = table.source.data.copy()
    new_data = table.source.data.copy()
    for key in new_data:
        new_data[key] = np.append(new_data[key], "")
    new_size = len(new_data[key])

    table.height = (new_size + 1) * 25
    table.source.data = new_data

    if size_toml_path is not None:
        nested_dict_set(control_data, size_toml_path, new_size - 1)


def show_hide_inputs(attr, old, new, linked_model_name: str, mode: str = "T") -> None:
    """
    Give the option to hide dependent inputs when the control input is set to "F" (False)
    This will help to only show pertinent information in the panels
    linked_model_name (str): name of the model which visibility depends on the value
    mode (str): the linked model visible attribute will be set to (new == mode)
    """
    linked_model = curdoc().select_one({"name": linked_model_name})
    linked_model.visible = new == mode


def checkboxgroup_update(
    attr,
    old,
    new,
    model_name: str,
    toml_control_path: str,
    as_bool: bool = False,
    linked_outputs: Optional[str] = None,
) -> None:
    """
    Update the control_data dictionary based on changes to a CheckboxGroup model

    model_name (str): name of the model that is triggering the update
    toml_control_path (str): dot-separated path to the variable to be updated in control_data
    as_bool (bool): if True, returns "T" for checked boxes and "F" for unchecked boxes
    """
    global control_data
    boxes = curdoc().select_one({"name": model_name})
    if as_bool:
        bool_result = ["T" if i in new else "F" for i, v in enumerate(boxes.labels)]
        if linked_outputs is None:
            nested_dict_set(control_data, toml_control_path, [bool_result[i] for i in new])
        else:
            linked_boxes = curdoc().select_one({"name": linked_outputs})
            nested_dict_set(
                control_data, toml_control_path, [bool_result[i] for i in linked_boxes.active]
            )
    else:
        nested_dict_set(control_data, toml_control_path, [boxes.labels[i] for i in new])


def radiogroup_update(attr, old, new, toml_control_path: str, one_based: bool = True) -> None:
    """
    Update the control_data dictionary based on changes to a RadioGroup or RadioButtonGroup model

    toml_control_path (str): dot-separated path to the variable to be updated in control_data
    """
    global control_data
    base = 1 if one_based else 0
    nested_dict_set(control_data, toml_control_path, new + base)


def update_visible(attr, old, new, target_model_name: str, model_name_dict: Dict[str, str]) -> None:
    """
    The visible attribute of target_model_name will be set based on the visible attribute of
    master_model_name and of the model that triggered the callback.

    target_model_name (str): name of the model which visible attribute will be updated
    model_name_dict (Dict[str,str]): keys are model names, values are in ["T","F",""]
            when values are "", a successive OR evaluation is done on the models values
            when values are ["T","F"], a successive AND evaluation is done on model_name_dict.values()
    """
    target_model = curdoc().select_one({"name": target_model_name})
    model_dict = {
        model_name: curdoc().select_one({"name": model_name}) for model_name in model_name_dict
    }

    mode = "OR" if ("" in model_name_dict.values()) else "AND"

    # OR starts at False, AND starts at True
    result = mode == "AND"
    for key, val in model_name_dict.items():
        if mode == "AND":
            result = result and (model_dict[key].value == val)
        elif mode == "OR":
            result = result or (model_dict[key].value == "T")

    target_model.visible = result


def update_visible_options(attr, old, new: int, model_name_list: Sequence[str]):
    """
    RadioGroup callback when the "active" attribute changes.
    When each option has an corresponding model, this makes visible only the model of the chosen option and hides the rest

    new (int): the updated 'active' attribute of the RadioGroup triggering the callback
    model_name_list (Sequence[str]): list of the model names that correspond to the RadioGroup labels (in order)
    """
    model_list = [curdoc().select_one({"name": model_name}) for model_name in model_name_list]
    for i, model in enumerate(model_list):
        model.visible = i == new


def update_control_field(
    attr, old, new, toml_control_path_list: Sequence[str], linked_value_model_list: Sequence[str]
) -> None:
    """
    Callback for models that update control_data when their value changes

    toml_control_path (Sequence[str]): list of dot-separated paths to the fields to be updated in control_data
    """
    global control_data

    if type(new) == np.str_:
        new = str(new)

    for toml_control_path in toml_control_path_list:
        nested_dict_set(control_data, toml_control_path, new)

    linked_model_list = [
        curdoc().select_one({"name": model_name}) for model_name in linked_value_model_list
    ]

    for model in linked_model_list:
        if hasattr(model, "value"):
            model.value = new
        elif hasattr(model, "source"):
            # harcode tables using this update to use new for the "file" column
            source_data = dict(model.source.data)
            if "file" in source_data:
                source_data["file"] = [new for i in source_data["file"]]
                model.source.data = source_data


def update_linked_tables(attr, old, new, linked_table_list: Sequence[str]) -> None:
    """
    When the table that triggers this callback is updated, update the list of linked tables to have the same number of rows.
    old and new are triggering_table.source.data

    linked_table_list (Optional[Sequence[str]]): list of names of table models to link to this table, they will always have the same number of rows as this table
    """

    linked_table_model_list = [
        curdoc().select_one({"name": linked_table_name}) for linked_table_name in linked_table_list
    ]

    old_data = dict(old)
    new_data = dict(new)

    # number of rows in the triggering table
    old_size = len(old_data[list(old_data.keys())[0]])
    new_size = len(new_data[list(new_data.keys())[0]])

    if (
        new_size != old_size
    ):  # only do things when the number of rows in the triggering table changes (and not when cell values change)
        for linked_table in linked_table_model_list:
            linked_table_data = dict(linked_table.source.data)
            linked_table_keys = list(linked_table_data.keys())

            if old_size > new_size:
                # remove lines in the linked table until it has new_size lines
                new_linked_table_data = {
                    k: linked_table_data[k] for k in linked_table_keys[: new_size - old_size]
                }
            elif old_size < new_size:
                # add lines in the linked table until it has new_size lines
                # copy the last filled line inputs for the new lines
                new_linked_table_data = {
                    k: np.array(list(v) + [v[-1] for i in range(new_size - old_size)])
                    for k, v in linked_table_data.items()
                }

            linked_table.source.data = new_linked_table_data

            linked_table.height = (new_size + 1) * 25


# Utility functions for generating models
def custom_div(text: str) -> bokeh.models.widgets.markups.Div:
    """
    Generate a Div with custom CSS
    """
    css = """
    div.bk-clearfix {
        font-weight: bold;
        color: teal;
    }
    """

    result = Div(text=text, stylesheets=[InlineStyleSheet(css=css)])

    return result


def build_model(
    model,
    toml_control_path_list: Sequence[str],
    is_bool: bool = False,
    linked_value_model_list: Sequence[str] = [],
    linked_model_list: Sequence[str] = [],
    linked_model_mode_list: Sequence[str] = [],
    value: Optional[Any] = None,
    **kwargs,
) -> Any:
    """
    Generic function to build a model with an on_change callback that assigns its value to the toml_control_path field in control_data
    model: any bokeh model function
    toml_control_path (Sequence[str]): list of dot-separated paths to the fields to be updated in control_data (including the present model first)
    is_bool (bool): use for Select widget that have "T"/"F" options to set their css update callback
    linked_value_model_list (Sequence[str]): list of model names with their value linked to the value of this model
    linked_model_list (Sequence[str]): list of model names with their visibility linked to the value of this model (only used when is_bool is True)
    linked_model_mode_list (Sequence[str]): "T"/"F" for each model in linked_model_list (only used when linked_model_list is not empty)
    """
    if value is None:
        value = nested_dict_get(control_data, toml_control_path_list[0])
        kwargs["value"] = value

    if type(kwargs["value"]) in [int, float]:
        # we use TextInput even for numeric input such that they can be used to give a template field instead of a value
        kwargs["value"] = str(kwargs["value"])

    result = model(**kwargs)
    result.on_change(
        "value",
        partial(
            update_control_field,
            toml_control_path_list=toml_control_path_list,
            linked_value_model_list=linked_value_model_list,
        ),
    )

    if is_bool:
        result.on_change("value", partial(select_status_update, model_name=kwargs["name"]))
        result.stylesheets = select_status_dict[
            nested_dict_get(control_data, toml_control_path_list[0])
        ]
        if linked_model_list and not linked_model_mode_list:
            linked_model_mode_list = ["T" for i in linked_model_list]
        for linked_model_name, mode in zip(linked_model_list, linked_model_mode_list):
            result.on_change(
                "value", partial(show_hide_inputs, linked_model_name=linked_model_name, mode=mode)
            )

    return result


def build_diag_model(
    title: str,
    toml_control_path: str,
    species_builder: Optional[Callable] = None,
) -> bokeh.models.layouts.Row:
    """
    Generic function to generate a repeated layout in the Diagnostic Options.
    It has two Select T/F widgets, for Prior and Posterior, and in some cases it allows specifying a list of species the option applies to.
    """
    name = toml_control_path.replace(".", "_")
    title_div = custom_div(text=title)
    toml_control_data = nested_dict_get(control_data, toml_control_path)
    selectors = [
        build_model(
            Select,
            toml_control_path_list=[f"{toml_control_path}.save_{elem}"],
            name=f"{name}_save_{elem}",
            title=f"Save {elem}",
            is_bool=True,
            options=["T", "F"],
            width=100,
        )
        for elem in ["prior", "posterior"]
    ]
    if (
        (species_builder is not None)
        or ("stokes" in toml_control_data)
        or ("reference_wavelength_nm" in toml_control_data)
    ):
        # set the callbacks to show the dependent inputs
        for elem in selectors:
            elem.on_change(
                "value",
                partial(
                    update_visible,
                    target_model_name=f"{name}_inputs",
                    model_name_dict={f"{name}_save_prior": "", f"{name}_save_posterior": ""},
                ),
            )

    selectors_row = row(children=selectors, width=800)

    selectors_inputs_children = []

    if species_builder is not None:
        default_active = toml_control_data["species"]
        if np.array_equal(toml_control_data["species"], ["all"]):
            if species_builder == gas_checkboxes:
                default_active = control_data["prf_species"] + control_data["col_species"]
            elif species_builder == aerosol_checkboxes:
                default_active = control_data["aerosol_species"]
            elif species_builder == cloud_checkboxes:
                default_active = control_data["cloud_species"]

        selectors_inputs_children += [
            species_builder(
                name=f"{name}_species",
                toml_control_path=f"{toml_control_path}.species",
                default_active=default_active,
                inline=True,
            )
        ]
    elif "species" in toml_control_data:
        selectors_inputs_children += [
            build_table(
                name=f"{name}_species",
                title="Species",
                toml_control_path=f"{toml_control_path}.species",
                fixed_column_width=100,
            )
        ]

    if "stokes" in toml_control_data:
        selectors_inputs_children += [
            build_model(
                Select,
                toml_control_path_list=[f"{toml_control_path}.stokes"],
                is_bool=True,
                options=["T", "F"],
                name=f"{name}_stokes",
                title="Stokes",
                width=100,
            )
        ]
    if "reference_wavelength_nm" in toml_control_data:
        selectors_inputs_children += [
            build_model(
                TextInput,
                toml_control_path_list=[f"{toml_control_path}.reference_wavelength_nm"],
                name=f"{name}_reference_wavelength",
                title="Reference Wavelength",
                width=200,
            )
        ]

    children = [title_div, selectors_row]
    if selectors_inputs_children:
        selectors_inputs = column(
            children=selectors_inputs_children,
            name=f"{name}_inputs",
            visible=(toml_control_data["save_prior"] == "T")
            or (toml_control_data["save_posterior"] == "T"),
        )

        children += [selectors_inputs]

    return column(children=children, name=name)


def build_inv_diag_model(
    title: str,
    toml_control_path: str,
    species_builder: Optional[Callable] = None,
) -> bokeh.models.layouts.Row:
    """
    Build a Select widget with the option of a linked checkboxgroup
    """

    name = toml_control_path.replace(".", "_")
    title_div = custom_div(text=title)
    toml_control_data = nested_dict_get(control_data, toml_control_path)

    linked_model_list = []
    if species_builder is not None:
        linked_model_list += [f"{name}_species"]

    selector = build_model(
        Select,
        toml_control_path_list=[f"{toml_control_path}.save"],
        name=f"{name}_save",
        is_bool=True,
        options=["T", "F"],
        linked_model_list=linked_model_list,
        width=100,
    )

    children = [title_div, selector]

    if species_builder is not None:
        default_active = toml_control_data["species"]
        if np.array_equal(toml_control_data["species"], ["all"]):
            if species_builder == gas_checkboxes:
                default_active = control_data["prf_species"] + control_data["col_species"]
            elif species_builder == aerosol_checkboxes:
                default_active = control_data["aerosol_species"]
            elif species_builder == cloud_checkboxes:
                default_active = control_data["cloud_species"]

        children += [
            species_builder(
                name=f"{name}_species",
                toml_control_path=f"{toml_control_path}.species",
                default_active=default_active,
                inline=True,
            )
        ]
        children[-1].visible = toml_control_data["save"] == "T"

    return column(children=children, name=name)


def build_table(
    name: str,
    data: Optional[Dict[str, Any]] = None,
    toml_control_path: Optional[str] = None,
    key_map: Optional[Dict[str, str]] = None,
    width_map: Optional[Dict[str, int]] = None,
    title: Optional[str] = None,
    add_button: bool = True,
    size_toml_path: Optional[str] = None,
    fixed_column_width: Optional[int] = None,
    mode: Optional[str] = None,
    direct_key: Optional[str] = None,
    width: int = 600,
    visible: bool = True,
    prune: Optional[Sequence[str]] = None,
    linked_table_list: Optional[Sequence[str]] = None,
) -> bokeh.models.layouts.Row:
    """
    Builds a table + add_row button for a given "data" dictionary

    name (str): name to use for the models
    data (Optional[Dict[str,Any]]): data dictionary to build the table (Mutually Inclusive: mode); when this is given without toml_control_path, there will not be an update callback
    toml_control_path (Optional[str]): dot-separated path to the data in control_data
    key_map (Optional[Dict[str,str]]): keys are the same as "data", values are the names of the columns
    width_map (Optional[Dict[str,int]]): keys are the same as "data", values are the width of the corresponding column in the table
    title (Optional[str]): if given, add a Div with this title above the table
    add_button (bool): if True, add an "add_row" button to the table
    size_toml_path (Optional[str]):dot-separated path in control_data to the argument that correspond to the number of rows of the table
        this will be used e.g. when there is a polynomial order, and then a table of uncertainties on coefficients, we will only have a table widget
        and it will update the polynomial order based on the number of rows in the uncertainty table
    fixed_column_width (Optional[int]): if given, all columns will have that width and the width argument will be overwritten to match
    mode (Optional[str]): one of ["direct","single_layer","double_layer"], must be specified when using the "data" arguments
    direct_key (Optional[str]): Sets the header name when updating a non-dictionary field, if not given, default to the "name" argument. Only used when mode!="direct"
    width (int): table width, will add +200 to have the add_row button to the right of the table when add_button==True
    visible (bool): sets the initial visibility of the returned layout
    prune (Optional[Sequence[str]]): list of keys to discard before generating the table
    linked_table_list (Optional[Sequence[str]]): list of names of table models to link to this table, they will always have the same number of rows as this table
    """
    if data is None and toml_control_path is None:
        raise UsageError(
            f"Model Name: {name}. build_table needs at least one of 'data' or 'toml_control_path' arguments"
        )

    if data is not None and mode is None:
        raise UsageError(
            f"Model Name: {name}. When 'data' is given, 'mode' argument must also be given when calling build_table"
        )

    if toml_control_path is not None and data is None:
        data_in = nested_dict_get(control_data, toml_control_path).copy()
        if prune is not None:
            for key in prune:
                if key in data_in:
                    del data_in[key]
        data_in_depth = dict_depth(data_in)
        if data_in_depth == 0:
            mode = "direct"
        elif data_in_depth == 1:
            mode = "single_layer"
        elif data_in_depth == 2:
            mode = "double_layer"
        else:
            raise NotImplementedError(
                f"Model Name: {name}. build_table is only implemented for single and double layered data, given layers: {data_in_depth}"
            )

        if mode == "direct":
            if direct_key is None:
                direct_key = name
            data = {direct_key: data_in}
        elif mode == "single_layer":
            data = {key: np.array([value]) for key, value in data_in.items()}
        elif mode == "double_layer":
            key_list = list(data_in.keys())
            subkey_list = list(data_in[key_list[0]].keys())
            data = {
                subkey: np.array([data_in[key][subkey] for key in key_list])
                for subkey in subkey_list
            }
    elif prune is not None:
        for key in prune:
            if key in data:
                del data[key]

    if key_map is None:
        key_map = {k: k for k, v in data.items()}

    source = ColumnDataSource(data=data, name=f"{name}_source")

    if fixed_column_width is not None:
        width_map = {k: fixed_column_width for k in data}
        width = fixed_column_width * len(list(data.keys()))

    if width_map is None:
        columns = [TableColumn(field=k, title=v) for k, v in key_map.items()]
    else:
        # special consideration for the "file" field to avoid tedious setups of width_map dictionaries when it is the only wider column
        if "file" in width_map and width_map["file"] < 300:
            width_map["file"] = 300
        columns = [TableColumn(field=k, title=v, width=width_map[k]) for k, v in key_map.items()]
    table = DataTable(
        source=source,
        columns=columns,
        editable=True,
        name=f"{name}_table",
        width=width,
        height=25 * (len(list(data.values())[0]) + 1),
    )
    if toml_control_path is not None:
        table.source.on_change(
            "data",
            partial(
                table_to_control, mode=mode, toml_control_path=toml_control_path, key_map=key_map
            ),
        )
    if linked_table_list is not None:
        table.source.on_change(
            "data",
            partial(update_linked_tables, linked_table_list=linked_table_list),
        )
    if add_button:
        button = Button(label="Add row", button_type="primary", name=f"{name}_button", width=100)
        button.on_click(partial(add_table_row, table=table, size_toml_path=size_toml_path))
    if title is not None:
        div_title = Div(text=title)
        table_column = column(children=[div_title, table])
    else:
        table_column = column(children=[table])

    if add_button:
        table_layout = row(
            children=[table_column, column(button)], name=name, width=width + 200, visible=visible
        )
    else:
        table_layout = row(children=[table_column], name=name, width=width, visible=visible)

    return table_layout


def make_radiogroup(
    name: str,
    toml_control_path: str,
    labels: Sequence[str],
    default_active: str,
    title: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Generic RadioGroup builder
    """
    if title is None:
        title = name.replace("_", " ")
    title_div = Div(text=title)
    group = RadioGroup(labels=labels, active=default_active, name=f"{name}_radiogroup", **kwargs)
    group.on_change("active", partial(radiogroup_update, toml_control_path=toml_control_path))

    return column(children=[title_div, group], width=200, name=name), group


def make_checkboxes(
    name: str,
    toml_control_path: str,
    labels: Sequence[str],
    default_active: Sequence[str] = [],
    as_bool: bool = False,
    linked_outputs: Optional[str] = None,
    **kwargs: Any,
) -> bokeh.models.layouts.Column:
    """
    Generic checkboxgroup builder
    """
    if np.array_equal(default_active, ["all"]):
        default_active = labels

    try:
        active = [labels.index(i) for i in default_active]
    except ValueError:
        print(f"One of the labels given to {name} are not in the default labels list")

    title_div = Div(text=name.replace("_", " "))
    boxes = CheckboxGroup(labels=labels, active=active, name=f"{name}_checkboxes", **kwargs)
    boxes.on_change(
        "active",
        partial(
            checkboxgroup_update,
            model_name=f"{name}_checkboxes",
            toml_control_path=toml_control_path,
            as_bool=as_bool,
            linked_outputs=linked_outputs,
        ),
    )

    return column(children=[title_div, boxes], width=200, name=name)


def gas_checkboxes(
    name: str,
    toml_control_path: str,
    default_active: Sequence[str] = [],
    species_list: Sequence[str] = gas_choices,
    **kwargs: Any,
) -> bokeh.models.layouts.Column:
    """
    Make a checkboxgroup to select for gases
    """

    return make_checkboxes(name, toml_control_path, species_list, default_active, **kwargs)


def aerosol_checkboxes(
    name: str,
    toml_control_path: str,
    default_active: Sequence[str] = [],
    species_list: Sequence[str] = aerosol_choices,
    **kwargs: Any,
) -> bokeh.models.layouts.Column:
    """
    Make a checkboxgroup to select for aerosols
    """

    return make_checkboxes(name, toml_control_path, species_list, default_active, **kwargs)


def cloud_checkboxes(
    name: str,
    toml_control_path: str,
    default_active: Sequence[str] = [],
    species_list: Sequence[str] = cloud_choices,
    **kwargs: Any,
) -> bokeh.models.layouts.Column:
    """
    Make a checkboxgroup to select for clouds
    """

    return make_checkboxes(name, toml_control_path, species_list, default_active, **kwargs)


# The following *_options functions each build a TabPanel object
# with the corresponding SPLAT control file options
def file_options() -> bokeh.models.layouts.TabPanel:
    """
    Generate the layouts in the Files panel
    """
    root_data_directory = build_model(
        TextInput,
        toml_control_path_list=["root_data_directory"],
        title="Root data directory",
        name="root_data_directory",
        width=1200,
    )

    log_file = build_model(
        TextInput,
        toml_control_path_list=["log_file"],
        title="Output Log file",
        name="log_file",
        width=1200,
    )

    output_file = build_model(
        TextInput,
        toml_control_path_list=["output_file"],
        title="Output Level2 file",
        name="output_file",
        width=1200,
    )
    l1_file = build_model(
        TextInput,
        toml_control_path_list=[
            "l1_file",
            "l2_2d_support_data.surface_altitude.file",
            "l2_2d_support_data.surface_winds.file",
            "l2_2d_support_data.chlorophyll.file",
            "l2_2d_support_data.ocean_salinity.file",
            "l2_2d_support_data.snow_parameters.file",
            "clouds.file",
        ]
        + [
            f"l1_radiance.band_inputs.{key}.file"
            for key in control_data["l1_radiance"]["band_inputs"].keys()
        ],
        linked_value_model_list=[
            "l2_2d_support_data_surface_altitude_file",
            "l2_2d_support_data_surface_winds_file",
            "l2_2d_support_data_chlorophyll_file",
            "l2_2d_support_data_ocean_salinity_file",
            "l2_2d_support_data_snow_parameters_file",
            "aux_lamb_clouds_file",
            "l1_radiance_band_inputs_table",
        ],
        title="Input Level1 file",
        name="l1_file",
        width=1200,
    )

    l2_met_file = build_model(
        TextInput,
        toml_control_path_list=["l2_met_file", "l2_profile_support_data.file"]
        + [
            f"l2_surface_reflectance.band_inputs.{key}.file"
            for key in control_data["l2_surface_reflectance"]["band_inputs"]
        ],
        linked_value_model_list=["l2_profile_support_data_file"]
        + ["l2_surface_reflectance_inputs_table"],
        title="Input a priori Level2 file",
        name="l2_met_file",
        width=1200,
    )

    info_div = Div(
        text="""
    <b>Notes:</b></br>
    Modifying the Input files here will update the corresponding fields in the other tabs.</br>
    """
    )

    file_inputs = column(
        children=[
            root_data_directory,
            log_file,
            output_file,
            l1_file,
            l2_met_file,
            info_div,
        ],
        name="file_inputs",
    )

    return TabPanel(title="Files", child=file_inputs, name="file_panel")


def retrieval_options() -> bokeh.models.layouts.TabPanel:
    """
    Generate the layouts in the Retrieval panel
    """
    calculation_mode = build_model(
        Select,
        toml_control_path_list=["calculation_mode"],
        title="Calculation Mode",
        options=["INVERSE", "FORWARD"],
        name="calculation_mode",
        width=300,
    )

    compute_all_pixels = build_model(
        Select,
        toml_control_path_list=["compute_all_pixels"],
        is_bool=True,
        title="Compute all pixels",
        options=["T", "F"],
        name="compute_all_pixels",
        width=200,
    )

    x_start = build_model(
        TextInput,
        toml_control_path_list=["x_start"],
        title="X start",
        name="x_start",
        width=150,
    )
    x_end = build_model(
        TextInput,
        toml_control_path_list=["x_end"],
        title="X end",
        name="x_end",
        width=150,
    )
    x_retrieval_range = row(children=[x_start, x_end])

    y_start = build_model(
        TextInput,
        toml_control_path_list=["y_start"],
        title="Y start",
        name="y_start",
        width=150,
    )
    y_end = build_model(
        TextInput,
        toml_control_path_list=["y_end"],
        title="Y end",
        name="y_end",
        width=150,
    )
    y_retrieval_range = row(children=[y_start, y_end])

    write_directly_to_file = build_model(
        Select,
        toml_control_path_list=["write_directly_to_file"],
        is_bool=True,
        title="Write directly to file",
        options=["T", "F"],
        name="write_directly_to_file",
        width=200,
    )

    use_file_lock = build_model(
        Select,
        toml_control_path_list=["use_file_lock"],
        is_bool=True,
        title="Use file lock",
        options=["T", "F"],
        name="use_file_lock",
        width=200,
    )

    overwrite_existing = build_model(
        Select,
        toml_control_path_list=["overwrite_existing"],
        is_bool=True,
        title="Overwrite existing",
        options=["T", "F"],
        name="overwrite_existing",
        width=200,
    )

    cache_pixel_output = build_model(
        Select,
        toml_control_path_list=["cache_pixel_output"],
        is_bool=True,
        title="Cache pixel output",
        options=["T", "F"],
        name="cache_pixel_output",
        width=200,
    )

    switch_on_debug = build_model(
        Select,
        toml_control_path_list=["switch_on_debug"],
        is_bool=True,
        title="Switch on debug",
        options=["T", "F"],
        name="switch_on_debug",
        width=200,
    )

    debug_level = build_model(
        Select,
        toml_control_path_list=["debug_level"],
        title="Debug level",
        options=["", "1", "2", "3"],
        name="debug_level",
        width=200,
    )

    retrieval_inputs = column(
        children=[
            calculation_mode,
            compute_all_pixels,
            x_retrieval_range,
            y_retrieval_range,
            write_directly_to_file,
            use_file_lock,
            overwrite_existing,
            cache_pixel_output,
            switch_on_debug,
            debug_level,
        ],
        name="retrieval_inputs",
    )

    return TabPanel(title="Retrieval", child=retrieval_inputs, name="retrieval_panel")


def show_single_pixel_calc_inputs(attr, old, new):
    """
    When do_single_pixel_calc value is "F" we can hide all
    the options that depend on it being set to "T"
    """
    single_pixel_calc_inputs = curdoc().select_one({"name": "single_pixel_calc_inputs"})
    single_pixel_calc_inputs.visible = new == "T"


def level1_options() -> bokeh.models.layouts.TabPanel:
    """
    Generate the layouts in the Level1 panel
    """
    do_single_pixel_calc = build_model(
        Select,
        toml_control_path_list=["do_single_pixel_calc_from_inp"],
        is_bool=True,
        linked_model_list=["single_pixel_calc_inputs", "l1_radiance_band_inputs"],
        linked_model_mode_list=["T", "F"],
        title="Do single pixel calculation from input",
        options=["T", "F"],
        name="do_single_pixel_calc",
        width=200,
    )

    # Band definition inputs
    single_pixel_calc_band_definition_template_data_map = {
        "name": "Band Name",
        "start_nm": "Band Start [nm]",
        "end_nm": "Band End [nm]",
        "sampling_nm": "Sampling [nm]",
        "view_dir_idx": "Viewing Direction Index",
    }

    single_pixel_band_definition = build_table(
        name="single_pixel_band_definition",
        toml_control_path="single_pixel_calc_inp.band_definition",
        key_map=single_pixel_calc_band_definition_template_data_map,
        title="Band Definition",
        width=1000,
    )
    # latitude corners
    single_pixel_calc_longitude_corner = build_table(
        name="single_pixel_calc_longitude_corner",
        toml_control_path="single_pixel_calc_inp.pixel_corners.longitudes_deg",
        title="Pixel Corner Longitude",
        add_button=False,
        width=500,
    )
    # longitude corners
    single_pixel_calc_latitude_corner = build_table(
        name="single_pixel_calc_latitude_corner",
        toml_control_path="single_pixel_calc_inp.pixel_corners.latitudes_deg",
        title="Pixel Corner Latitude",
        add_button=False,
        width=500,
    )
    # Other single_pixel_calc inputs
    single_pixel_calc_template_data_map = {
        "date_yyyymmdd": "Date [YYYYMMDD]",
        "hour_decimal_utc": "Hour [decimal UTC]",
        "sza_deg": "SZA [deg]",
        "vza_deg": "VZA [deg]",
        "aza_deg": "AZA [deg]",
        "saa_deg": "SAA [deg]",
        "vaa_deg": "VAA [deg]",
        "longitude_deg": "Longitude [deg]",
        "latitude_deg": "Latitude [deg]",
        "surface_altitude_km": "Surface Altitude [km]",
        "obs_altitude_km": "Obs. Altitude [km]",
        "cloud_fraction": "Cloud Fraction",
        "cloud_pressure_hpa": "Cloud Pressure [hPa]",
        "wind_speed_mps": "Wind Speed [m/s]",
        "wind_dir_deg": "Wind Dir.[deg,c/w frm N]",
        "chlorophyll_mgpm3": "Chlorphyll [mg/m3]",
        "ocean_salinity_pptv": "Ocean Salinity [pptv]",
        "snow_fraction": "Snow Fraction",
        "sea_ice_fraction": "Sea Ice Fraction",
        "snow_depth_m": "Snow Depth [m]",
        "snow_age_days": "Snow Age [days]",
    }
    single_pixel_calc = build_table(
        name="single_pixel_calc",
        toml_control_path="single_pixel_calc_inp",
        key_map=single_pixel_calc_template_data_map,
        title="Single pixel calculation inputs",
        add_button=False,
        width=1400,
        prune=["band_definition", "pixel_corners"],
    )

    # puts all the single pixel_calc models together
    single_pixel_calc_inputs = column(
        children=[
            single_pixel_band_definition,
            single_pixel_calc_longitude_corner,
            single_pixel_calc_latitude_corner,
            single_pixel_calc,
        ],
        name="single_pixel_calc_inputs",
        visible=control_data["do_single_pixel_calc_from_inp"] == "T",
        width=1500,
    )

    l1_radiance_template_data_map = {
        "name": "Band Name",
        "index": "Band Index",
        "file_type": "File Type",
        "file": "File",
    }
    l1_radiance_band_inputs = build_table(
        name="l1_radiance_band_inputs",
        toml_control_path="l1_radiance.band_inputs",
        title="L1 Radiance Band Inputs",
        key_map=l1_radiance_template_data_map,
        width_map={"name": 100, "file_type": 100, "index": 100, "file": 1100},
        visible=control_data["do_single_pixel_calc_from_inp"] == "F",
        width=1400,
    )

    # L2 2D Support Data
    l2_d2_support_data_models = [custom_div(text="Level2 2D Support Data")]
    for key in control_data["l2_2d_support_data"]:
        nice_key = key.replace("_", " ")
        select_use = build_model(
            Select,
            toml_control_path_list=[f"l2_2d_support_data.{key}.use"],
            is_bool=True,
            linked_model_list=[f"l2_2d_support_data_{key}_inputs"],
            title=f"Use {nice_key} 2D Support Data",
            options=["T", "F"],
            name=f"use_l2_2d_support_data_{key}",
            width=200,
            stylesheets=select_status_dict[control_data["l2_2d_support_data"][key]["use"]],
        )

        select_file_type = build_model(
            Select,
            toml_control_path_list=[f"l2_2d_support_data.{key}.file_type"],
            title=f"{nice_key} File Type",
            options=["SPLAT", "METHANESAT"],
            name=f"l2_2d_support_data_{key}_file_type",
            width=200,
        )

        select_file = build_model(
            TextInput,
            toml_control_path_list=[f"l2_2d_support_data.{key}.file"],
            title=f"{nice_key} File",
            name=f"l2_2d_support_data_{key}_file",
            width=1200,
        )

        select_inputs = column(
            children=[select_file_type, select_file],
            name=f"l2_2d_support_data_{key}_inputs",
            visible=control_data["l2_2d_support_data"][key]["use"] == "T",
        )

        l2_d2_support_data_models += [select_use, select_inputs]

    # L2 Profile Support Data
    use_l2_profile_met = build_model(
        Select,
        toml_control_path_list=["l2_profile_support_data.use_l2_profile_met"],
        is_bool=True,
        linked_model_list=["l2_profile_support_data_inputs"],
        title="Use Level2 Profile Meteorology",
        options=["T", "F"],
        name="use_l2_profile_met",
        width=200,
    )

    l2_profile_support_data_file_type = build_model(
        Select,
        toml_control_path_list=["l2_profile_support_data.file_type"],
        title="Level2 Meteorology File Type",
        options=["SPLAT", "METHANESAT"],
        name="l2_profile_support_data_file_type",
        width=200,
    )

    l2_profile_support_data_file = build_model(
        TextInput,
        toml_control_path_list=["l2_profile_support_data.file"],
        title="Level2 Meteorology File",
        name="l2_profile_support_data_file",
        width=1200,
    )

    l2_profile_support_data_gases = gas_checkboxes(
        name="l2_profile_support_data_gases",
        toml_control_path="l2_profile_support_data.gases",
        default_active=control_data["l2_profile_support_data"]["gases"],
    )

    l2_profile_support_data_aerosols = aerosol_checkboxes(
        name="l2_profile_support_data_aerosols",
        toml_control_path="l2_profile_support_data.aerosols",
        default_active=control_data["l2_profile_support_data"]["aerosols"],
    )

    l2_profile_support_data_gases_and_aerosols = row(
        children=[column(l2_profile_support_data_gases), column(l2_profile_support_data_aerosols)]
    )

    l2_profile_support_data_inputs = column(
        children=[
            l2_profile_support_data_file_type,
            l2_profile_support_data_file,
            l2_profile_support_data_gases_and_aerosols,
        ],
        name="l2_profile_support_data_inputs",
        width=1500,
    )

    # L2 clouds
    use_l2_clouds = build_model(
        Select,
        toml_control_path_list=["clouds.use_l2_clouds"],
        is_bool=True,
        linked_model_list=["cloud_inputs"],
        title="Use Level2 Clouds",
        options=["T", "F"],
        name="use_l2_clouds",
        width=200,
    )
    use_l2_clouds.on_change(
        "value",
        partial(
            update_visible,
            target_model_name="use_clouds_from_l2_prof_met_species",
            model_name_dict={"use_l2_clouds": "T", "use_clouds_from_l2_prof_met": "T"},
        ),
    )
    use_l2_clouds.on_change(
        "value",
        partial(
            update_visible,
            target_model_name="aux_lamb_clouds_inputs",
            model_name_dict={"use_l2_clouds": "T", "use_clouds_from_l2_prof_met": "F"},
        ),
    )

    use_clouds_from_l2_prof_met = build_model(
        Select,
        toml_control_path_list=["clouds.use_clouds_from_l2_prof_met"],
        is_bool=True,
        title="Use Clouds from Level2 Profile Meteorology",
        options=["T", "F"],
        name="use_clouds_from_l2_prof_met",
        width=200,
    )
    use_clouds_from_l2_prof_met.on_change(
        "value",
        partial(
            update_visible,
            target_model_name="use_clouds_from_l2_prof_met_species",
            model_name_dict={"use_l2_clouds": "T", "use_clouds_from_l2_prof_met": "T"},
        ),
    )
    use_clouds_from_l2_prof_met.on_change(
        "value",
        partial(
            update_visible,
            target_model_name="aux_lamb_clouds_inputs",
            model_name_dict={"use_l2_clouds": "T", "use_clouds_from_l2_prof_met": "F"},
        ),
    )

    use_clouds_from_l2_prof_met_species = cloud_checkboxes(
        name="use_clouds_from_l2_prof_met_species",
        toml_control_path="clouds.species",
        default_active=control_data["clouds"]["species"],
    )
    use_clouds_from_l2_prof_met_species.visible = (
        control_data["clouds"]["use_clouds_from_l2_prof_met"] == "T"
    ) and (control_data["clouds"]["use_l2_clouds"] == "T")

    aux_lamb_clouds_file_type = build_model(
        Select,
        toml_control_path_list=["clouds.file_type"],
        title="Aux Lambertian Cloud File Type",
        options=["SPLAT", "METHANESAT"],
        name="aux_lamb_clouds_file_type",
        width=200,
    )

    aux_lamb_clouds_file = build_model(
        TextInput,
        toml_control_path_list=["clouds.file"],
        title="Aux Lambertian Cloud File",
        name="aux_lamb_clouds_file",
        width=1200,
    )

    aux_lamb_clouds_inputs = column(
        children=[aux_lamb_clouds_file_type, aux_lamb_clouds_file],
        visible=control_data["clouds"]["use_clouds_from_l2_prof_met"] == "F"
        and (control_data["clouds"]["use_l2_clouds"] == "T"),
        name="aux_lamb_clouds_inputs",
    )

    cloud_inputs = column(
        name="cloud_inputs",
        children=[
            use_clouds_from_l2_prof_met,
            use_clouds_from_l2_prof_met_species,
            aux_lamb_clouds_inputs,
        ],
        visible=control_data["clouds"]["use_l2_clouds"] == "T",
    )

    # L2 Surface Reflectance
    use_l2_surface = build_model(
        Select,
        toml_control_path_list=["l2_surface_reflectance.use_l2_surface"],
        is_bool=True,
        linked_model_list=["l2_surface_reflectance_inputs"],
        title="Use Level2 Surface Reflectance",
        options=["T", "F"],
        name="use_l2_surface",
        width=200,
    )

    l2_surface_reflectance_template_data_map = {
        "name": "Band Name",
        "index": "Band Index",
        "file_type": "File Type",
        "file": "File",
    }
    l2_surface_reflectance_width_map = {
        "name": 100,
        "index": 100,
        "file_type": 100,
        "file": 1100,
    }
    l2_surface_reflectance_inputs = build_table(
        name="l2_surface_reflectance_inputs",
        toml_control_path="l2_surface_reflectance.band_inputs",
        key_map=l2_surface_reflectance_template_data_map,
        width_map=l2_surface_reflectance_width_map,
        title="L2 Surface Reflectance",
        width=1400,
    )

    level1_inputs_models = (
        [do_single_pixel_calc, single_pixel_calc_inputs, l1_radiance_band_inputs]
        + l2_d2_support_data_models
        + [use_l2_profile_met, l2_profile_support_data_inputs]
        + [use_l2_clouds, cloud_inputs]
        + [use_l2_surface, l2_surface_reflectance_inputs]
    )

    level1_inputs = column(
        children=level1_inputs_models,
        name="level1_inputs",
        width=1600,
    )

    return TabPanel(title="Level1", child=level1_inputs, name="level1_panel")


def rtm_options() -> bokeh.models.layouts.TabPanel:
    """
    Generate the layouts in the RTM panel
    """

    vlidort_line_by_line_inputs = build_table(
        name="vlidort_line_by_line_inputs",
        toml_control_path="vlidort_line_by_line",
        fixed_column_width=100,
        title="VLIDORT Line-By-Line",
    )

    vlidort_pca_inputs = build_table(
        name="vlidort_pca_inputs",
        toml_control_path="vlidort_pca",
        fixed_column_width=100,
        title="VLIDORT PCA",
    )

    first_order_inputs = build_table(
        name="first_order_inputs",
        toml_control_path="first_order",
        fixed_column_width=100,
        title="First Order",
    )

    two_stream_inputs = build_table(
        name="two_stream_inputs",
        toml_control_path="two_stream",
        fixed_column_width=100,
        title="Two Stream",
    )

    rtm_inputs = column(
        children=[
            vlidort_line_by_line_inputs,
            vlidort_pca_inputs,
            first_order_inputs,
            two_stream_inputs,
        ],
        name="rtm_inputs",
        width=1600,
    )
    return TabPanel(title="RTM", child=rtm_inputs, name="rtm_panel")


def window_options() -> bokeh.models.layouts.TabPanel:
    """
    Generate the layouts in the Window panel
    """
    isrf_parametrization_div = custom_div(text="ISRF Parametrizations")

    fixed_supergaussian_inputs = build_table(
        name="fixed_supergaussian_inputs",
        toml_control_path="isrf_parametrization.fixed_supergaussian",
        fixed_column_width=100,
        title="Fixed Supergaussian",
    )

    fixed_tropomi_inputs = build_table(
        name="fixed_tropomi_inputs",
        toml_control_path="isrf_parametrization.fixed_tropomi",
        fixed_column_width=100,
        title="Fixed TROPOMI",
        width=800,
    )

    isrf_file_inputs = build_table(
        name="isrf_file_inputs",
        toml_control_path="isrf_parametrization.isrf_file",
        fixed_column_width=100,
        title="ISRF Input File (will look for the file under {{root_data_directory}}/../../)",
        width=800,
    )

    isrf_parametrization_inputs = column(
        children=[
            isrf_parametrization_div,
            fixed_supergaussian_inputs,
            fixed_tropomi_inputs,
            isrf_file_inputs,
        ],
        name="isrf_parametrization_inputs",
    )

    amf_mode_options_inputs = build_table(
        name="amf_mode_options_inputs",
        toml_control_path="amf_mode_options",
        title="AMF Mode Options",
        add_button=False,
        width=600,
    )

    fwd_inv_mode_options_width_map = {
        "index": 100,
        "start_nm": 100,
        "end_nm": 100,
        "buffer_nm": 100,
        "convolution_grid_sampling": 200,
        "radiative_transfer_model": 200,
        "isrf": 100,
        "convolution_width_hw1e": 200,
        "use_fft_for_convolution": 200,
    }
    fwd_inv_mode_options_inputs = build_table(
        name="fwd_inv_mode_options_inputs",
        toml_control_path="fwd_inv_mode_options",
        width_map=fwd_inv_mode_options_width_map,
        title="Forward/Inverse Mode Options",
        width=1300,
        # the following tables all depend on the number of windows set by the fwd_inv_mode_options_inputs table
        linked_table_list=[
            "surface_reflectance_window_poly_scale_inputs_table",
            "radiometric_offset_window_poly_scale_inputs_table",
            "radiometric_scaling_window_poly_scale_inputs_table",
            "wavelength_grid_window_poly_scale_inputs_table",
        ],
    )

    common_options_div = custom_div(text="Common Options")

    solar_reference_file = build_model(
        TextInput,
        toml_control_path_list=["common_options.solar_file"],
        title="Solar Reference (will look under {{root_data_directory}}/SolarSpectra/)",
        name="solar_reference_file",
        width=1200,
    )

    rtm_at_l1_resolution = build_model(
        Select,
        toml_control_path_list=["common_options.rtm_at_l1_resolution"],
        is_bool=True,
        title="RTM at L1 Resolution",
        options=["T", "F"],
        name="rtm_at_l1_resolution",
        width=200,
    )
    rtm_at_l1_resolution.on_change(
        "value",
        partial(
            update_visible,
            target_model_name="solar_io_correction",
            model_name_dict={"rtm_at_l1_resolution": "T"},
        ),
    )
    rtm_at_l1_resolution.on_change(
        "value",
        partial(
            update_visible,
            target_model_name="use_custom_rtm_grid",
            model_name_dict={"rtm_at_l1_resolution": "F"},
        ),
    )
    rtm_at_l1_resolution.on_change(
        "value",
        partial(
            update_visible,
            target_model_name="custom_grid_file",
            model_name_dict={"rtm_at_l1_resolution": "F", "use_custom_rtm_grid": "T"},
        ),
    )

    solar_io_correction = build_model(
        Select,
        toml_control_path_list=["common_options.solar_io_correction"],
        is_bool=True,
        title="Solar I0 Correction",
        options=["T", "F"],
        name="solar_io_correction",
        width=200,
        visible=control_data["common_options"]["rtm_at_l1_resolution"] == "T",
    )

    use_custom_rtm_grid = build_model(
        Select,
        toml_control_path_list=["common_options.use_custom_rtm_grid"],
        is_bool=True,
        linked_model_list=["custom_grid_file"],
        title="Use Custom RTM grid",
        options=["T", "F"],
        name="use_custom_rtm_grid",
        width=200,
        visible=control_data["common_options"]["rtm_at_l1_resolution"] == "F",
    )

    custom_grid_file = build_model(
        TextInput,
        toml_control_path_list=["common_options.custom_grid_file"],
        name="custom_grid_file",
        title="Custom Grid Filename (will look under {{root_data_directory}}/../../)",
        visible=(control_data["common_options"]["use_custom_rtm_grid"] == "T")
        and (control_data["common_options"]["rtm_at_l1_resolution"] == "F"),
        width=1200,
    )

    rtm_at_l1_resolution_inputs = column(
        children=[solar_io_correction, use_custom_rtm_grid, custom_grid_file],
        name="rtm_at_l1_resolution_inputs",
    )

    common_options_inputs = column(
        children=[
            common_options_div,
            solar_reference_file,
            rtm_at_l1_resolution,
            rtm_at_l1_resolution_inputs,
        ],
        name="common_options_inputs",
    )

    window_inputs = column(
        children=[
            isrf_parametrization_inputs,
            amf_mode_options_inputs,
            fwd_inv_mode_options_inputs,
            common_options_inputs,
        ],
        name="window_inputs",
        width=1600,
    )
    return TabPanel(title="Window", child=window_inputs, name="window_panel")


def profile_options() -> bokeh.models.layouts.TabPanel:
    """
    Generate the layouts in the Profile panel
    """
    apriori_file = build_model(
        TextInput,
        toml_control_path_list=["apriori_file"],
        name="apriori_file",
        title="Level2 a priori filename (will look under {{root_data_directory}}/../../)",
        width=1200,
    )

    sampling_method, _ = make_radiogroup(
        name="sampling_method",
        toml_control_path="sampling_method",
        labels=["Nearest Neighbor", "Point-in-Polygon"],
        default_active=control_data["sampling_method"] - 1,
    )

    assume_earth_for_gravity = build_model(
        Select,
        toml_control_path_list=["assume_earth_for_gravity"],
        is_bool=True,
        linked_model_list=["assume_earth_for_gravity_inputs"],
        linked_model_mode_list=["F"],
        title="Assume Earth for Gravity",
        options=["T", "F"],
        name="assume_earth_for_gravity",
        width=200,
        stylesheets=select_status_dict[control_data["assume_earth_for_gravity"]],
    )

    surface_gravity = build_model(
        TextInput,
        toml_control_path_list=["surface_gravity_mps2"],
        name="surface_gravity",
        title="Surface Gravity (m.s-2)",
        width=200,
    )

    planetary_radius = build_model(
        TextInput,
        toml_control_path_list=["planetary_radius_km"],
        name="planetary_radius",
        title="Planetary Radius (km)",
        width=200,
    )

    assume_earth_for_gravity_inputs = column(
        children=[surface_gravity, planetary_radius],
        name="assume_earth_for_gravity_inputs",
        visible=control_data["assume_earth_for_gravity"] == "F",
    )

    linearize_wrt_hybrid_grid = build_model(
        Select,
        toml_control_path_list=["linearize_wrt_hybrid_grid"],
        is_bool=True,
        title="Linearize w.r.t. Hybrid Grid",
        options=["T", "F"],
        name="linearize_wrt_hybrid_grid",
        width=200,
    )

    # Profile Trace Gases
    profile_trace_gases = gas_checkboxes(
        name="profile_trace_gases",
        toml_control_path="profile_trace_gases",
        default_active=control_data["profile_trace_gases"],
        inline=True,
    )

    wet_trace_gases = gas_checkboxes(
        name="wet_trace_gases",
        toml_control_path="wet_trace_gases",
        default_active=control_data["wet_trace_gases"],
        inline=True,
    )

    proxy_normalization_species = gas_checkboxes(
        name="proxy_normalization_species",
        toml_control_path="proxy_normalization_species",
        default_active=control_data["proxy_normalization_species"],
        inline=True,
    )

    trace_gas_properties_inputs = build_table(
        name="trace_gas_properties_inputs",
        toml_control_path="trace_gas_properties",
        fixed_column_width=100,
        title="Trace Gas Properties",
        width=300,
    )

    # Profile Aerosols
    aerosol_species = aerosol_checkboxes(
        name="aerosol_species",
        toml_control_path="aerosol_species",
        default_active=control_data["aerosol_species"],
        inline=True,
    )

    aod_reference_wavelength = build_model(
        TextInput,
        toml_control_path_list=["profile_aerosols.aod_reference_wavelength_nm"],
        name="aod_reference_wavelength",
        title="AOD Reference Wavelength (nm)",
        width=200,
    )

    aerosol_params = aerosol_checkboxes(
        name="aerosol_params",
        toml_control_path="aerosol_params",
        species_list=aerosol_param_choices,
        default_active=control_data["aerosol_params"],
        inline=True,
    )

    aod_from_profile_file = build_model(
        Select,
        toml_control_path_list=["profile_aerosols.aod_from_profile_file"],
        is_bool=True,
        linked_model_list=["profile_aer_opt_prop_params_inputs"],
        linked_model_mode_list=["F"],
        title="AOD from Profile File",
        options=["T", "F"],
        name="aod_from_profile_file",
        width=200,
    )

    profile_aer_opt_prop_params_inputs = build_table(
        name="profile_aer_opt_prop_params_inputs",
        toml_control_path="profile_aerosols.profile_aer_opt_prop_params",
        fixed_column_width=100,
        title="Aerosol Profile Parameters",
        visible=control_data["profile_aerosols"]["aod_from_profile_file"] == "F",
    )

    aer_opt_par_from_profile_file = build_model(
        Select,
        toml_control_path_list=["profile_aerosols.aer_opt_par_from_profile_file"],
        is_bool=True,
        linked_model_list=["const_values_inputs"],
        linked_model_mode_list=["F"],
        title="Aerosol Optical Parameters from Profile File",
        options=["T", "F"],
        name="aer_opt_par_from_profile_file",
        width=300,
    )

    const_values_inputs = build_table(
        name="const_values_inputs",
        toml_control_path="profile_aerosols.const_values",
        fixed_column_width=100,
        title="Aerosol Optical Parameters",
        visible=control_data["profile_aerosols"]["aer_opt_par_from_profile_file"] == "F",
    )

    profile_aerosol_inputs = column(
        children=[
            aerosol_species,
            aod_reference_wavelength,
            aerosol_params,
            aod_from_profile_file,
            profile_aer_opt_prop_params_inputs,
            aer_opt_par_from_profile_file,
            const_values_inputs,
        ],
        name="profile_aerosol_inputs",
    )

    # Profile cloud species
    cloud_species = cloud_checkboxes(
        name="cloud_species",
        toml_control_path="cloud_species",
        default_active=control_data["cloud_species"],
        inline=True,
    )

    cod_reference_wavelength = build_model(
        TextInput,
        toml_control_path_list=["profile_cloud_species.cod_reference_wavelength_nm"],
        name="cod_reference_wavelength",
        title="COD Reference Wavelength (nm)",
        width=200,
    )

    cod_from_profile_file = build_model(
        Select,
        toml_control_path_list=["profile_cloud_species.cod_from_profile_file"],
        is_bool=True,
        linked_model_list=["cod_from_profile_file_inputs"],
        linked_model_mode_list=["F"],
        title="COD from Profile File",
        options=["T", "F"],
        name="cod_from_profile_file",
        width=200,
        stylesheets=select_status_dict[
            control_data["profile_cloud_species"]["cod_from_profile_file"]
        ],
    )

    n_subpixels = build_model(
        TextInput,
        toml_control_path_list=["profile_cloud_species.n_subpixels"],
        name="n_subpixels",
        title="# Subpixels",
        width=200,
    )

    cloud_fractions = build_table(
        name="cloud_fractions",
        toml_control_path="profile_cloud_species.cloud_fractions",
        direct_key="Cloud Fractions",
        title="Cloud Fractions",
        add_button=False,
        width=200,
    )

    cloud_pressure_hpa = build_table(
        name="cloud_pressure_hpa",
        toml_control_path="profile_cloud_species.cloud_pressure_hpa",
        direct_key="Cloud Pressure (hPa)",
        title="Cloud Pressure",
        add_button=False,
        width=200,
    )

    cloud_profile_type_inputs = build_table(
        name="cloud_profile_type_inputs",
        toml_control_path="profile_cloud_species.species_proftype",
        fixed_column_width=100,
        title="Cloud species",
    )

    cloud_profile_params_inputs = build_table(
        name="cloud_profile_params_inputs",
        toml_control_path="profile_cloud_species.profile_params",
        fixed_column_width=100,
        title="Cloud Parameters",
    )

    cod_from_profile_file_inputs = column(
        children=[
            cloud_species,
            n_subpixels,
            cloud_fractions,
            cloud_pressure_hpa,
            cloud_profile_type_inputs,
            cloud_profile_params_inputs,
        ],
        name="cod_from_profile_file_inputs",
        visible=control_data["profile_cloud_species"]["cod_from_profile_file"] == "F",
    )

    profile_cloud_inputs = column(
        children=[
            cod_reference_wavelength,
            cod_from_profile_file,
            cod_from_profile_file_inputs,
        ],
        name="profile_cloud_inputs",
    )

    profile_inputs = column(
        children=[
            apriori_file,
            sampling_method,
            assume_earth_for_gravity,
            assume_earth_for_gravity_inputs,
            linearize_wrt_hybrid_grid,
            profile_trace_gases,
            wet_trace_gases,
            proxy_normalization_species,
            trace_gas_properties_inputs,
            profile_aerosol_inputs,
            profile_cloud_inputs,
        ],
        name="profile_inputs",
        width=1600,
    )

    return TabPanel(title="Profile", child=profile_inputs, name="profile_panel")


def surface_options() -> bokeh.models.layouts.TabPanel:
    """
    Generate the layouts in the Surface panel
    """

    surface_reflectance_option, surface_reflectance_option_radiogroup = make_radiogroup(
        name="surface_reflectance_option",
        toml_control_path="surface_reflectance_option",
        labels=[
            "Fixed Lambertian",
            "Lambertian Spectrum",
            "LER Climatology",
            "MODIS-FA",
            "Fixed Kernel BRDF",
            "BRDF Climatology",
        ],
        default_active=control_data["surface_reflectance_option"] - 1,
        title="Surface Reflectance Option",
    )
    surface_reflectance_option_radiogroup.on_change(
        "active",
        partial(
            update_visible_options,
            model_name_list=[
                "fixed_lambertian_albedo",
                "lambertian_surface_file",
                "ler_climatology_inputs",
                "modis_fa_inputs",
                "fixed_kernel_brdf_inputs",
                "brdf_climatology_file",
            ],
        ),
    )

    fixed_lambertian_albedo = build_model(
        TextInput,
        toml_control_path_list=["surface_reflectance_options.fixed_lambertian.albedo_value"],
        name="fixed_lambertian_albedo",
        title="Fixed Lambertian Albedo",
        width=200,
        visible=surface_reflectance_option_radiogroup.active == 0,
    )

    lambertian_surface_file = build_model(
        TextInput,
        toml_control_path_list=[
            "surface_reflectance_options.lambertian_spectrum.lambertian_surface_file"
        ],
        name="lambertian_surface_file",
        title="Lambertian Surface Filename (will look for the file under {{root_data_directory}}/ReflSpectra/)",
        visible=surface_reflectance_option_radiogroup.active == 1,
        width=1200,
    )

    ler_use_constant_wavelength = build_model(
        Select,
        toml_control_path_list=[
            "surface_reflectance_options.ler_climatology.use_constant_wavelength"
        ],
        is_bool=True,
        linked_model_list=["ler_wavelength"],
        title="Use Constant Wavelength",
        options=["T", "F"],
        name="ler_use_constant_wavelength",
        width=200,
    )

    ler_wavelength = build_model(
        TextInput,
        toml_control_path_list=["surface_reflectance_options.ler_climatology.ler_wavelength_nm"],
        name="ler_wavelength",
        title="LER Wavelength (nm)",
        width=200,
        visible=control_data["surface_reflectance_options"]["ler_climatology"][
            "use_constant_wavelength"
        ]
        == "T",
    )

    ler_climatology_file = build_model(
        TextInput,
        toml_control_path_list=["surface_reflectance_options.ler_climatology.ler_climatology_file"],
        name="ler_climatology_file",
        title="LER Climatology File (will look under {{root_data_directory}}/LER_climatologies/)",
        width=1200,
    )

    ler_climatology_inputs = column(
        children=[ler_use_constant_wavelength, ler_wavelength, ler_climatology_file],
        name="ler_climatology_inputs",
        visible=surface_reflectance_option_radiogroup.active == 2,
    )

    modis_fa_file = build_model(
        TextInput,
        toml_control_path_list=["surface_reflectance_options.modis_fa.modis_fa_file"],
        name="modis_fa_file",
        title="MODIS-FA Filename (will look under {{root_data_directory}}/BRDF_EOF/AlbSpec/)",
        width=1200,
    )

    modis_fa_refl_clim_directory = build_model(
        TextInput,
        toml_control_path_list=["surface_reflectance_options.modis_fa.refl_clim_directory"],
        name="modis_fa_refl_clim_directory",
        title="Reflectance Climatology Directory (under {{root_data_directory}})",
        width=1200,
    )

    modis_fa_do_isotropic = build_model(
        Select,
        toml_control_path_list=["surface_reflectance_options.modis_fa.do_isotropic"],
        is_bool=True,
        linked_model_list=["modis_fa_black_white_blue", "modis_fa_ocean_glint_brdf"],
        linked_model_mode_list=["T", "F"],
        title="Do Isotropic",
        options=["T", "F"],
        name="modis_fa_do_isotropic",
        width=200,
    )

    modis_fa_black_white_blue, _ = make_radiogroup(
        name="modis_fa_black_white_blue",
        toml_control_path="surface_reflectance_options.modis_fa.black_white_blue",
        labels=["Black", "White", "Blue"],
        default_active=control_data["surface_reflectance_options"]["modis_fa"]["black_white_blue"]
        - 1,
        visible=control_data["surface_reflectance_options"]["modis_fa"]["do_isotropic"] == "T",
        title="Isotropic Option",
    )

    modis_fa_ocean_glint_brdf = build_model(
        Select,
        toml_control_path_list=["surface_reflectance_options.modis_fa.ocean_glint_brdf"],
        is_bool=True,
        title="Ocean Glint BRDF",
        options=["T", "F"],
        name="modis_fa_ocean_glint_brdf",
        width=200,
    )

    modis_fa_inputs = column(
        children=[
            modis_fa_file,
            modis_fa_refl_clim_directory,
            modis_fa_do_isotropic,
            modis_fa_black_white_blue,
            modis_fa_ocean_glint_brdf,
        ],
        name="modis_fa_inputs",
        visible=surface_reflectance_option_radiogroup.active == 3,
    )

    fixed_kernel_brdf_inputs = build_table(
        name="fixed_kernel_brdf_inputs",
        toml_control_path="surface_reflectance_options.fixed_kernel_brdf.vlidort_options",
        fixed_column_width=100,
        title="Fixed Kernel BRDF VLIDORT Options",
        visible=surface_reflectance_option_radiogroup.active == 4,
    )

    brdf_climatology_file = build_model(
        TextInput,
        toml_control_path_list=[
            "surface_reflectance_options.brdf_climatology.brdf_climatology_file"
        ],
        name="brdf_climatology_file",
        title="BRDF Climatology File",
        visible=surface_reflectance_option_radiogroup.active == 5,
        width=1200,
    )

    surface_emissivity_option, surface_emissivity_option_radiogroup = make_radiogroup(
        name="surface_emissivity_option",
        toml_control_path="surface_emissivity_option",
        labels=[
            "Fixed Emissivity",
            "Emissivity Spectrum",
            "Emissivity Climatology",
        ],
        default_active=control_data["surface_emissivity_option"] - 1,
        title="Surface Emissivity Option",
    )
    surface_emissivity_option_radiogroup.on_change(
        "active",
        partial(
            update_visible_options,
            model_name_list=[
                "fixed_emissivity_value",
                "emissivity_spectrum_file",
                "emissivity_climatology_file",
            ],
        ),
    )

    fixed_emissivity_value = build_model(
        TextInput,
        toml_control_path_list=["surface_emissivity_options.fixed_emissivity.emissivity_value"],
        name="fixed_emissivity_value",
        title="Fixed Emissivity Value",
        width=200,
        visible=surface_emissivity_option_radiogroup.active == 0,
    )

    emissivity_spectrum_file = build_model(
        TextInput,
        toml_control_path_list=[
            "surface_emissivity_options.emissivity_spectrum.emissivity_spectrum_file"
        ],
        name="emissivity_spectrum_file",
        title="Emissivity Spectrum Filename (will look under {{root_data_directory}}/Emissivity)",
        visible=surface_emissivity_option_radiogroup.active == 1,
        width=1200,
    )

    emissivity_climatology_file = build_model(
        TextInput,
        toml_control_path_list=[
            "surface_emissivity_options.emissivity_climatology.emissivity_climatology_file"
        ],
        name="emissivity_climatology_file",
        title="Emissivity Climatology Filename (will look under {{root_data_directory}}/Emissivity)",
        visible=surface_emissivity_option_radiogroup.active == 2,
        width=1200,
    )

    do_plant_fluorescence = build_model(
        Select,
        toml_control_path_list=["do_plant_fluorescence"],
        is_bool=True,
        linked_model_list=["chlorophyll_spectrum_file"],
        title="Do Plant Fluorescence",
        options=["T", "F"],
        name="do_plant_fluorescence",
        width=200,
        stylesheets=select_status_dict[control_data["do_plant_fluorescence"]],
    )

    chlorophyll_spectrum_file = build_model(
        TextInput,
        toml_control_path_list=["chlorophyll_spectrum_file"],
        name="chlorophyll_spectrum_file",
        title="Chlorophyll Spectrum Filename",
        visible=control_data["do_plant_fluorescence"] == "T",
        width=1200,
    )

    surface_inputs = column(
        children=[
            surface_reflectance_option,
            fixed_lambertian_albedo,
            lambertian_surface_file,
            ler_climatology_inputs,
            modis_fa_inputs,
            fixed_kernel_brdf_inputs,
            brdf_climatology_file,
            surface_emissivity_option,
            fixed_emissivity_value,
            emissivity_spectrum_file,
            emissivity_climatology_file,
            do_plant_fluorescence,
            chlorophyll_spectrum_file,
        ],
        name="surface_inputs",
        width=1600,
    )

    return TabPanel(title="Surface", child=surface_inputs, name="surface_panel")


def gas_options() -> bokeh.models.layouts.TabPanel:
    """
    Generate the layouts in the Gas panel
    """

    assume_earth_for_scattering = build_model(
        Select,
        toml_control_path_list=["assume_earth_for_scattering"],
        is_bool=True,
        linked_model_list=["co2_profile_gas_name", "scattering_gases_inputs"],
        linked_model_mode_list=["T", "F"],
        title="Assume Earth for Scattering",
        options=["T", "F"],
        name="assume_earth_for_scattering",
        width=200,
    )

    co2_profile_gas_name = build_model(
        TextInput,
        toml_control_path_list=["co2_profile_gas_name"],
        name="co2_profile_gas_name",
        title="CO2 Profile Gas Name",
        width=200,
        visible=control_data["assume_earth_for_scattering"] == "T",
    )

    scattering_gases = gas_checkboxes(
        name="scattering_gases",
        toml_control_path="scattering_gases",
        default_active=control_data["scattering_gases"],
        inline=True,
    )

    scattering_gas_entries_inputs = build_table(
        name="scattering_gas_entries_inputs",
        toml_control_path="scattering_gas_entries",
        fixed_column_width=100,
        title="Scattering Gas Entries (will look for files under {{root_data_directory}}/GasScattering)",
    )

    scattering_gases_inputs = column(
        children=[scattering_gases, scattering_gas_entries_inputs],
        name="scattering_gases_inputs",
        visible=control_data["assume_earth_for_scattering"] == "F",
    )

    do_raman_scattering = build_model(
        Select,
        toml_control_path_list=["do_raman_scattering"],
        is_bool=True,
        linked_model_list=["raman_scattering_inputs"],
        title="Do Raman Scattering",
        options=["T", "F"],
        name="do_raman_scattering",
        width=200,
        stylesheets=select_status_dict[control_data["do_raman_scattering"]],
    )

    rss_ref_temperature = build_model(
        TextInput,
        toml_control_path_list=["rss_ref_temperature_k"],
        name="rss_ref_temperature",
        title="RSS Reference Temperature (K)",
        width=200,
    )

    raman_gases = gas_checkboxes(
        name="raman_gases",
        toml_control_path="raman_gases",
        default_active=control_data["raman_gases"],
        inline=True,
    )

    raman_gas_entries_inputs = build_table(
        name="raman_gas_entries_inputs",
        toml_control_path="raman_gas_entries",
        fixed_column_width=100,
        title="Raman Gas Entries (will look for files under {{root_data_directory}}/GasScattering)",
    )

    raman_scattering_inputs = column(
        children=[rss_ref_temperature, raman_gases, raman_gas_entries_inputs],
        name="raman_scattering_inputs",
        visible=control_data["do_raman_scattering"] == "T",
    )

    absorbing_gases = gas_checkboxes(
        name="absorbing_gases",
        toml_control_path="abs_species",
        default_active=control_data["abs_species"],
        species_list=["N2", "O2", "Ar", "H2O", "CH4", "CO2", "PACIA"],
        inline=True,
    )

    number_of_fine_layers = build_model(
        TextInput,
        toml_control_path_list=["number_of_fine_layers"],
        name="number_of_fine_layers",
        title="Number of Fine Layers for Cross Section Calculations",
        width=300,
    )

    full_lut_to_memory = build_model(
        Select,
        toml_control_path_list=["full_lut_to_memory"],
        is_bool=True,
        title="Full Lookup-Table to Memory",
        options=["T", "F"],
        name="full_lut_to_memory",
        width=200,
        stylesheets=select_status_dict[control_data["full_lut_to_memory"]],
    )

    cross_section_entries_inputs = build_table(
        name="cross_section_entries_inputs",
        toml_control_path="cross_section_entries",
        fixed_column_width=100,
        title="Cross-Section Entries (will look for files under {{root_data_directory}}/SAO_crosssections/splatv2_xsect)",
        width=800,
    )

    airglow_gases = gas_checkboxes(
        name="airglow_gases",
        toml_control_path="airglow_gases",
        species_list=["O2DG"],
        default_active=control_data["airglow_gases"],
        inline=True,
    )

    airglow_cross_section_entries_inputs = build_table(
        name="airglow_cross_section_entries_inputs",
        toml_control_path="airglow_cross_section_entries",
        fixed_column_width=100,
        title="Airglow Cross-Section Entries (will look for files under {{root_data_directory}}/SAO_crosssections/splatv2_xsect)",
        width=800,
    )

    gas_inputs = column(
        children=[
            assume_earth_for_scattering,
            co2_profile_gas_name,
            scattering_gases_inputs,
            do_raman_scattering,
            raman_scattering_inputs,
            absorbing_gases,
            number_of_fine_layers,
            full_lut_to_memory,
            cross_section_entries_inputs,
            airglow_gases,
            airglow_cross_section_entries_inputs,
        ],
        name="gas_inputs",
        width=1600,
    )
    return TabPanel(title="Gas", child=gas_inputs, name="gas_panel")


def aerosol_options() -> bokeh.models.layouts.TabPanel:
    """
    Generate the layouts in the Aerosol panel
    """

    do_aerosols = build_model(
        Select,
        toml_control_path_list=["do_aerosols"],
        is_bool=True,
        title="Do Aerosols",
        options=["T", "F"],
        name="do_aerosols",
        width=200,
    )

    aerosol_optical_properties_inputs = build_table(
        name="aerosol_optical_properties_inputs",
        toml_control_path="aerosol_optical_properties",
        fixed_column_width=100,
        title="Cross-Section Entries (will look for files under {{root_data_directory}}/AerCldProp)",
        width=800,
    )

    aerosol_inputs = column(
        children=[
            do_aerosols,
            aerosol_optical_properties_inputs,
        ],
        name="aerosol_inputs",
        width=1600,
    )

    return TabPanel(title="Aerosol", child=aerosol_inputs, name="aerosol_panel")


def cloud_options() -> bokeh.models.layouts.TabPanel:
    """
    Generate the layouts in the Cloud panel
    """
    do_clouds = build_model(
        Select,
        toml_control_path_list=["do_clouds"],
        is_bool=True,
        linked_model_list=["do_cloud_inputs"],
        title="Do Clouds",
        options=["T", "F"],
        name="do_clouds",
        width=200,
    )

    do_lambertian_clouds = build_model(
        Select,
        toml_control_path_list=["do_lambertian_clouds"],
        is_bool=True,
        linked_model_list=["cloud_albedo", "cloud_opt_prop_inputs"],
        linked_model_mode_list=["T", "F"],
        title="Do Lambertian Clouds",
        options=["T", "F"],
        name="do_lambertian_clouds",
        width=200,
    )

    cloud_albedo = build_model(
        TextInput,
        toml_control_path_list=["cloud_albedo"],
        name="cloud_albedo",
        title="Cloud Albedo",
        width=200,
        visible=control_data["do_lambertian_clouds"] == "T",
    )

    cloud_opt_prop_inputs = build_table(
        name="cloud_opt_prop_inputs",
        toml_control_path="cloud_opt_prop",
        fixed_column_width=100,
        title="Cloud Optical Properties (will look for files under {{root_data_directory}}/AerCldProp)",
        width=800,
        visible=control_data["do_lambertian_clouds"] == "F",
    )

    do_cloud_inputs = column(
        children=[do_lambertian_clouds, cloud_albedo, cloud_opt_prop_inputs],
        name="do_cloud_inputs",
        visible=control_data["do_clouds"] == "T",
    )

    cloud_options_inputs = column(
        children=[
            do_clouds,
            do_cloud_inputs,
        ],
        name="cloud_options_inputs",
        width=1600,
    )

    return TabPanel(title="Cloud", child=cloud_options_inputs, name="cloud_panel")


def state_vector_options() -> bokeh.models.layouts.TabPanel:
    """
    Generate the layouts in the State Vector panel
    """
    # Trace Gases
    trace_gas_div = custom_div(text="Trace Gases")
    profile_species = gas_checkboxes(
        name="profile_species",
        toml_control_path="prf_species",
        default_active=control_data["prf_species"],
        inline=True,
    )

    column_species = gas_checkboxes(
        name="column_species",
        toml_control_path="col_species",
        default_active=control_data["col_species"],
        inline=True,
    )

    overwrite_clim_uncertainty = build_model(
        Select,
        toml_control_path_list=["overwrite_clim_uncertainty"],
        is_bool=True,
        linked_model_list=["clim_uncertainty_options_inputs"],
        title="Overwrite Climatology Uncertainty",
        options=["T", "F"],
        name="overwrite_clim_uncertainty",
        width=200,
    )

    clim_uncertainty_options_inputs = build_table(
        name="clim_uncertainty_options_inputs",
        toml_control_path="uncertainty_option",
        fixed_column_width=100,
        title="Climatology Uncertainty Options",
        width=400,
        visible=control_data["overwrite_clim_uncertainty"] == "T",
    )

    trace_gas_inputs = column(
        children=[
            trace_gas_div,
            profile_species,
            column_species,
            overwrite_clim_uncertainty,
            clim_uncertainty_options_inputs,
        ],
        name="trace_gas_inputs",
    )

    # Aerosols
    aerosol_div = custom_div(text="Aerosol")
    aod_species = aerosol_checkboxes(
        name="aod_species",
        toml_control_path="aerosol.aod_species",
        species_list=aerosol_choices,
        default_active=control_data["aerosol"]["aod_species"],
        inline=True,
    )
    aod_species_inputs = build_table(
        name="aod_species_param",
        toml_control_path="aerosol.aod_species_param",
        fixed_column_width=100,
        title="Profile Param. Species AOD",
        width=300,
    )
    aerosol_opt_prop_species = aerosol_checkboxes(
        name="aerosol_opt_prop_species",
        toml_control_path="aerosol.opt_prop_species",
        species_list=aerosol_choices,
        default_active=control_data["aerosol"]["opt_prop_species"],
        inline=True,
    )

    aerosol_inputs = column(
        children=[aerosol_div, aod_species, aod_species_inputs, aerosol_opt_prop_species],
        name="state_vector_aerosol_inputs",
    )

    # Temperature
    temperature_div = custom_div(text="Temperature")
    add_temperature = build_model(
        Select,
        toml_control_path_list=["temperature.add_to_state_vector"],
        is_bool=True,
        linked_model_list=["add_temperature_inputs"],
        title="Add Temperature to State Vector",
        options=["T", "F"],
        name="add_temperature",
        width=200,
    )

    temperature_option, temperature_option_radiogroup = make_radiogroup(
        name="temperature_option",
        toml_control_path="temperature.temperature_option",
        labels=["Fit Profile", "Fit Shift"],
        default_active=control_data["temperature"]["temperature_option"] - 1,
        title="Temperature Option",
    )
    temperature_option_radiogroup.on_change(
        "active",
        partial(
            update_visible_options,
            model_name_list=[
                "fit_profile_inputs",
                "fit_shift_inputs",
            ],
        ),
    )

    fit_profile_overwrite_clim_uncertainty = build_model(
        Select,
        toml_control_path_list=["temperature.fit_profile.overwrite_clim_uncertainty"],
        is_bool=True,
        linked_model_list=["fit_profile_overwrite_clim_uncertainty_inputs"],
        title="Overwrite Climatology Uncertainty",
        options=["T", "F"],
        name="fit_profile_overwrite_clim_uncertainty",
        width=200,
    )

    fit_profile_covar_matrix_type = build_model(
        TextInput,
        toml_control_path_list=["temperature.fit_profile.cov_mat_type"],
        name="fit_profile_covar_matrix_type",
        title="Covariance Matrix Type",
        width=200,
    )

    fit_profile_parameters = build_model(
        TextInput,
        toml_control_path_list=["temperature.fit_profile.parameters"],
        name="fit_profile_parameters",
        title="Parameters",
        width=200,
    )

    fit_profile_overwrite_clim_uncertainty_inputs = column(
        children=[fit_profile_covar_matrix_type, fit_profile_parameters],
        name="fit_profile_overwrite_clim_uncertainty_inputs",
        visible=control_data["temperature"]["fit_profile"]["overwrite_clim_uncertainty"] == "T",
    )

    fit_profile_inputs = column(
        children=[
            fit_profile_overwrite_clim_uncertainty,
            fit_profile_overwrite_clim_uncertainty_inputs,
        ],
        name="fit_profile_inputs",
        visible=control_data["temperature"]["temperature_option"] == 1,
    )

    fit_shift_overwrite_clim_uncertainty = build_model(
        Select,
        toml_control_path_list=["temperature.fit_shift.overwrite_clim_uncertainty"],
        is_bool=True,
        linked_model_list=["fit_shift_uncertainty"],
        title="Overwrite Climatology Uncertainty",
        options=["T", "F"],
        name="fit_shift_overwrite_clim_uncertainty",
        width=200,
    )

    fit_shift_uncertainty = build_model(
        TextInput,
        toml_control_path_list=["temperature.fit_shift.shift_unc_k"],
        name="fit_shift_uncertainty",
        title="Shift Uncertainty (K)",
        width=200,
        visible=control_data["temperature"]["fit_shift"]["overwrite_clim_uncertainty"] == "T",
    )

    fit_shift_inputs = column(
        children=[fit_shift_overwrite_clim_uncertainty, fit_shift_uncertainty],
        name="fit_shift_inputs",
        visible=control_data["temperature"]["temperature_option"] == 2,
    )

    add_temperature_inputs = column(
        children=[temperature_option, fit_profile_inputs, fit_shift_inputs],
        name="add_temperature_inputs",
        visible=control_data["temperature"]["add_to_state_vector"] == "T",
    )

    temperature_inputs = column(
        children=[temperature_div, add_temperature, add_temperature_inputs],
        name="temperature_inputs",
    )

    # Surface Pressure
    surface_pressure_div = custom_div(text="Surface Pressure")
    add_surface_pressure = build_model(
        Select,
        toml_control_path_list=["surface_pressure.add_to_state_vector"],
        is_bool=True,
        linked_model_list=["add_surface_pressure_inputs"],
        title="Add Surface Pressure to State Vector",
        options=["T", "F"],
        name="add_surface_pressure",
        width=200,
    )

    surface_pressure_overwrite_clim_uncertainty = build_model(
        Select,
        toml_control_path_list=["surface_pressure.overwrite_clim_uncertainty"],
        is_bool=True,
        linked_model_list=["surface_pressure_uncertainty"],
        title="Overwrite Climatology Uncertainty",
        options=["T", "F"],
        name="surface_pressure_overwrite_clim_uncertainty",
        width=200,
    )

    surface_pressure_uncertainty = build_model(
        TextInput,
        toml_control_path_list=["surface_pressure.uncertainty_hpa"],
        name="surface_pressure_uncertainty",
        title="Surface Pressure Uncertainty (hPa)",
        width=200,
        visible=control_data["surface_pressure"]["overwrite_clim_uncertainty"] == "T",
    )

    add_surface_pressure_inputs = column(
        children=[
            surface_pressure_overwrite_clim_uncertainty,
            surface_pressure_uncertainty,
        ],
        name="add_surface_pressure_inputs",
        visible=control_data["surface_pressure"]["add_to_state_vector"] == "T",
    )

    surface_pressure_inputs = column(
        children=[
            surface_pressure_div,
            add_surface_pressure,
            add_surface_pressure_inputs,
        ],
        name="surface_pressure_inputs",
    )

    # Surface Reflectance
    surface_reflectance_div = custom_div(text="Surface Reflectance")
    add_surface_reflectance = build_model(
        Select,
        toml_control_path_list=["surface_reflectance.add_to_state_vector"],
        is_bool=True,
        linked_model_list=["add_surface_reflectance_inputs"],
        title="Add Surface Reflectance to State Vector",
        options=["T", "F"],
        name="add_surface_reflectance",
        width=200,
    )

    init_state_from_obs = build_model(
        Select,
        toml_control_path_list=["surface_reflectance.init_state_from_obs"],
        is_bool=True,
        linked_model_list=["init_state_from_obs_inputs"],
        title="Initialize State from Observation",
        options=["T", "F"],
        name="init_state_from_obs",
        width=200,
    )

    albedo_file = build_model(
        TextInput,
        toml_control_path_list=["surface_reflectance.albedo_file"],
        name="albedo_file",
        title="Albedo File (will look under {{root_data_directory}}/../../)",
        width=1200,
    )

    max_poly_order = build_model(
        TextInput,
        toml_control_path_list=["surface_reflectance.max_poly_order"],
        name="max_poly_order",
        title="Maximum Polynomial Order",
        width=200,
    )

    surface_reflectance_fit_option, surface_reflectance_fit_option_radiogroup = make_radiogroup(
        name="surface_reflectance_fit_option",
        toml_control_path="surface_reflectance.option_index",
        labels=["Polynomial Scaling", "Window Poly. Scale", "EOF fit"],
        default_active=control_data["surface_reflectance"]["option_index"] - 1,
        title="Polynomial Fit Option",
    )
    surface_reflectance_fit_option_radiogroup.on_change(
        "active",
        partial(
            update_visible_options,
            model_name_list=[
                "surface_reflectance_polynomial_scaling_inputs",
                "surface_reflectance_window_poly_scale_inputs",
                "surface_reflectance_eof_fit_inputs",
            ],
        ),
    )

    surface_reflectance_polynomial_scaling_inputs = build_table(
        name="surface_reflectance_polynomial_scaling_inputs",
        title="Polynomial Scaling Uncertainty (%)",
        toml_control_path="surface_reflectance.polynomial_scaling.uncert_prcnt",
        direct_key="Coeff Uncertainty (%)",
        width=200,
        size_toml_path="surface_reflectance.polynomial_scaling.order",
        visible=control_data["surface_reflectance"]["option_index"] == 1,
    )

    surface_reflectance_window_poly_scale_inputs = build_table(
        name="surface_reflectance_window_poly_scale_inputs",
        title="Window Polynomial Scaling (one row per fitting window)",
        toml_control_path="surface_reflectance.window_poly_scale",
        add_button=False,  # number of rows in this table depends on fwd_inv_mode_options_inputs
        width=400,
        visible=control_data["surface_reflectance"]["option_index"] == 2,
    )

    surface_reflectance_scale_eof_uncert = build_model(
        Select,
        toml_control_path_list=["surface_reflectance.eof_fit.scale_eof_uncert"],
        is_bool=True,
        linked_model_list=["surface_reflectance_eof_scale_factor"],
        title="Scale EOF Uncertainty",
        options=["T", "F"],
        name="surface_reflectance_scale_eof_uncert",
        width=200,
    )

    surface_reflectance_eof_scale_factor = build_model(
        TextInput,
        toml_control_path_list=["surface_reflectance.eof_fit.scale_factor"],
        name="surface_reflectance_eof_scale_factor",
        title="EOF Uncertainty Scale Factor",
        width=200,
        visible=control_data["surface_reflectance"]["eof_fit"]["scale_eof_uncert"] == "T",
    )

    surface_reflectance_eof_fit_inputs = column(
        children=[surface_reflectance_scale_eof_uncert, surface_reflectance_eof_scale_factor],
        name="surface_reflectance_eof_fit_inputs",
        visible=control_data["surface_reflectance"]["option_index"] == 3,
    )

    init_state_from_obs_inputs = column(
        children=[
            albedo_file,
            max_poly_order,
            surface_reflectance_fit_option,
            surface_reflectance_polynomial_scaling_inputs,
            surface_reflectance_window_poly_scale_inputs,
            surface_reflectance_eof_fit_inputs,
        ],
        name="init_state_from_obs_inputs",
        visible=control_data["surface_reflectance"]["init_state_from_obs"] == "T",
    )

    add_surface_reflectance_inputs = column(
        children=[init_state_from_obs, init_state_from_obs_inputs],
        name="add_surface_reflectance_inputs",
        visible=control_data["surface_reflectance"]["add_to_state_vector"] == "T",
    )

    surface_reflectance_inputs = column(
        children=[
            surface_reflectance_div,
            add_surface_reflectance,
            add_surface_reflectance_inputs,
        ],
        name="surface_reflectance_inputs",
    )

    # Radiometric Offset
    radiometric_offset_div = custom_div(text="Radiometric Offset")
    add_radiometric_offset = build_model(
        Select,
        toml_control_path_list=["radiometric_offset.add_to_state_vector"],
        is_bool=True,
        linked_model_list=["add_radiometric_offset_inputs"],
        title="Add Radiometric Offset to State Vector",
        options=["T", "F"],
        name="add_radiometric_offset",
        width=200,
    )

    radiometric_offset_fit_option, radiometric_offset_fit_option_radiogroup = make_radiogroup(
        name="radiometric_offset_fit_option",
        toml_control_path="radiometric_offset.option_index",
        labels=["Polynomial Scaling", "Window Poly. Scale"],
        default_active=control_data["radiometric_offset"]["option_index"] - 1,
        title="Polynomial Fit Option",
    )
    radiometric_offset_fit_option_radiogroup.on_change(
        "active",
        partial(
            update_visible_options,
            model_name_list=[
                "radiometric_offset_polynomial_scaling_inputs",
                "radiometric_offset_window_poly_scale_inputs",
            ],
        ),
    )

    radiometric_offset_polynomial_scaling_inputs = build_table(
        name="radiometric_offset_polynomial_scaling_inputs",
        title="Polynomial Scaling Uncertainty (%)",
        toml_control_path="radiometric_offset.polynomial_scaling.uncert_prcnt",
        direct_key="Coeff Uncertainty (%)",
        width=200,
        size_toml_path="radiometric_offset.polynomial_scaling.order",
        visible=control_data["radiometric_offset"]["option_index"] == 1,
    )

    radiometric_offset_window_poly_scale_inputs = build_table(
        name="radiometric_offset_window_poly_scale_inputs",
        title="Window Polynomial Scaling (one row per fitting window)",
        toml_control_path="radiometric_offset.window_poly_scale",
        add_button=False,  # number of rows in this table depends on fwd_inv_mode_options_inputs
        width=400,
        visible=control_data["radiometric_offset"]["option_index"] == 2,
    )

    add_radiometric_offset_inputs = column(
        children=[
            radiometric_offset_fit_option,
            radiometric_offset_polynomial_scaling_inputs,
            radiometric_offset_window_poly_scale_inputs,
        ],
        name="add_radiometric_offset_inputs",
        visible=control_data["radiometric_offset"]["add_to_state_vector"] == "T",
    )

    radiometric_offset_inputs = column(
        children=[
            radiometric_offset_div,
            add_radiometric_offset,
            add_radiometric_offset_inputs,
        ],
        name="radiometric_offset_inputs",
    )

    # Radiometric Scaling
    radiometric_scaling_div = custom_div(text="Radiometric Scaling")
    add_radiometric_scaling = build_model(
        Select,
        toml_control_path_list=["radiometric_scaling.add_to_state_vector"],
        is_bool=True,
        linked_model_list=["add_radiometric_scaling_inputs"],
        title="Add Radiometric Scaling to State Vector",
        options=["T", "F"],
        name="add_radiometric_scaling",
        width=200,
    )

    radiometric_scaling_fit_option, radiometric_scaling_fit_option_radiogroup = make_radiogroup(
        name="radiometric_scaling_fit_option",
        toml_control_path="radiometric_scaling.option_index",
        labels=["Polynomial Scaling", "Window Poly. Scale"],
        default_active=control_data["radiometric_scaling"]["option_index"] - 1,
        title="Polynomial Fit Option",
    )
    radiometric_offset_fit_option_radiogroup.on_change(
        "active",
        partial(
            update_visible_options,
            model_name_list=[
                "radiometric_offset_polynomial_scaling_inputs",
                "radiometric_offset_window_poly_scale_inputs",
            ],
        ),
    )

    radiometric_scaling_polynomial_scaling_inputs = build_table(
        name="radiometric_scaling_polynomial_scaling_inputs",
        title="Polynomial Scaling Uncertainty (%)",
        toml_control_path="radiometric_scaling.polynomial_scaling.uncert_prcnt",
        direct_key="Coeff Uncertainty (%)",
        width=200,
        size_toml_path="radiometric_scaling.polynomial_scaling.order",
        visible=control_data["radiometric_scaling"]["option_index"] == 1,
    )

    radiometric_scaling_window_poly_scale_inputs = build_table(
        name="radiometric_scaling_window_poly_scale_inputs",
        title="Window Polynomial Scaling (one row per fitting window)",
        toml_control_path="radiometric_scaling.window_poly_scale",
        add_button=False,  # number of rows in this table depends on fwd_inv_mode_options_inputs
        width=400,
    )

    add_radiometric_scaling_inputs = column(
        children=[
            radiometric_scaling_fit_option,
            radiometric_scaling_polynomial_scaling_inputs,
            radiometric_scaling_window_poly_scale_inputs,
        ],
        name="add_radiometric_scaling_inputs",
        visible=control_data["radiometric_scaling"]["add_to_state_vector"] == "T",
    )

    radiometric_scaling_inputs = column(
        children=[
            radiometric_scaling_div,
            add_radiometric_scaling,
            add_radiometric_scaling_inputs,
        ],
        name="radiometric_scaling_inputs",
    )

    # EOF residuals
    eof_residuals_div = custom_div(text="EOF Residuals")
    add_eof_residuals = build_model(
        Select,
        toml_control_path_list=["eof_residuals.add_to_state_vector"],
        is_bool=True,
        linked_model_list=["add_eof_residuals_inputs"],
        title="Add EOF Residuals to State Vector",
        options=["T", "F"],
        name="add_eof_residuals",
        width=200,
    )

    add_eof_residuals_inputs = build_table(
        name="add_eof_residuals_inputs",
        toml_control_path="eof_residuals.eofs_to_add",
        title="EOFs to add",
        visible=control_data["eof_residuals"]["add_to_state_vector"] == "T",
    )

    eof_residuals_inputs = column(
        children=[eof_residuals_div, add_eof_residuals, add_eof_residuals_inputs],
        name="eof_residuals_inputs",
    )

    # ISRF Parameters
    isrf_parameters_div = custom_div(text="ISRF Parameters")

    add_isrf_parameters = build_model(
        Select,
        toml_control_path_list=["isrf_parameters.add_to_state_vector"],
        is_bool=True,
        linked_model_list=["add_isrf_parameters_inputs"],
        title="Add ISRF Parameters to State Vector",
        options=["T", "F"],
        name="add_isrf_parameters",
        width=200,
    )

    add_isrf_parameters_inputs = build_table(
        name="add_isrf_parameters_inputs",
        toml_control_path="isrf_parameters.parameters_to_fit",
        title="Parameters to fit",
        width=500,
        visible=control_data["isrf_parameters"]["add_to_state_vector"] == "T",
    )

    isrf_parameters_inputs = column(
        children=[isrf_parameters_div, add_isrf_parameters, add_isrf_parameters_inputs],
        name="isrf_parameters_inputs",
    )

    # Wavelength Grid
    wavelength_grid_div = custom_div(text="Wavelength Grid")
    add_wavelength_grid = build_model(
        Select,
        toml_control_path_list=["wavelength_grid.add_to_state_vector"],
        is_bool=True,
        linked_model_list=["add_wavelength_grid_inputs"],
        title="Add Wavelength Grid to State Vector",
        options=["T", "F"],
        name="add_wavelength_grid",
        width=200,
    )

    wavelength_grid_fit_option, wavelength_grid_fit_option_radiogroup = make_radiogroup(
        name="wavelength_grid_fit_option",
        toml_control_path="wavelength_grid.option_index",
        labels=["Polynomial Scaling", "Window Poly. Scale"],
        default_active=control_data["wavelength_grid"]["option_index"] - 1,
        title="Polynomial Fit Option",
    )
    wavelength_grid_fit_option_radiogroup.on_change(
        "active",
        partial(
            update_visible_options,
            model_name_list=[
                "wavelength_grid_polynomial_scaling_inputs",
                "wavelength_grid_window_poly_scale_inputs",
            ],
        ),
    )

    wavelength_grid_polynomial_scaling_inputs = build_table(
        name="wavelength_grid_polynomial_scaling_inputs",
        title="Polynomial Scaling Uncertainty (%)",
        toml_control_path="wavelength_grid.polynomial_scaling.uncert_prcnt",
        direct_key="Coeff Uncertainty (%)",
        width=200,
        size_toml_path="wavelength_grid.polynomial_scaling.order",
        visible=control_data["wavelength_grid"]["option_index"] == 1,
    )

    wavelength_grid_window_poly_scale_inputs = build_table(
        name="wavelength_grid_window_poly_scale_inputs",
        title="Window Polynomial Scaling (one row per fitting window)",
        toml_control_path="wavelength_grid.window_poly_scale",
        add_button=False,  # number of rows in this table depends on fwd_inv_mode_options_inputs
        width=400,
        visible=control_data["wavelength_grid"]["option_index"] == 2,
    )

    add_wavelength_grid_inputs = column(
        children=[
            wavelength_grid_fit_option,
            wavelength_grid_polynomial_scaling_inputs,
            wavelength_grid_window_poly_scale_inputs,
        ],
        name="add_wavelength_grid_inputs",
        visible=control_data["wavelength_grid"]["add_to_state_vector"] == "T",
    )

    wavelength_grid_inputs = column(
        children=[
            wavelength_grid_div,
            add_wavelength_grid,
            add_wavelength_grid_inputs,
        ],
        name="wavelength_grid_inputs",
    )

    # initialize wavelengths from cross correlation
    init_wvl_div = custom_div(text="Initialize Wavelengths")
    init_wvl = build_model(
        Select,
        toml_control_path_list=["init_wavelength_from_cross_correlation.init_wvl"],
        is_bool=True,
        linked_model_list=["max_pixel_shift"],
        title="Init wavelengths from cross correlation",
        options=["T", "F"],
        name="init_wvl",
        width=200,
    )

    max_pixel_shift = build_model(
        TextInput,
        toml_control_path_list=["init_wavelength_from_cross_correlation.max_pixel_shift"],
        name="max_pixel_shift",
        title="Max. pixel shift (pixels)",
        width=200,
        visible=control_data["init_wavelength_from_cross_correlation"]["init_wvl"] == "T",
    )

    init_wavelength_inputs = column(
        children=[
            init_wvl_div,
            init_wvl,
            max_pixel_shift,
        ],
        name="init_wvl_inputs",
    )

    state_vector_inputs = column(
        children=[
            trace_gas_inputs,
            aerosol_inputs,
            temperature_inputs,
            surface_pressure_inputs,
            surface_reflectance_inputs,
            radiometric_offset_inputs,
            radiometric_scaling_inputs,
            eof_residuals_inputs,
            isrf_parameters_inputs,
            wavelength_grid_inputs,
            init_wavelength_inputs,
        ],
        name="state_vector_inputs",
        width=1600,
    )

    return TabPanel(title="State Vector", child=state_vector_inputs, name="state_vector_panel")


def optimizer_options() -> bokeh.models.layouts.TabPanel:
    """
    Generate the layouts in the Optimizer panel
    """
    max_iter = build_model(
        TextInput,
        toml_control_path_list=["max_iter"],
        name="max_iter",
        title="Maximum # of Iterations",
        width=200,
    )

    max_div = build_model(
        TextInput,
        toml_control_path_list=["max_div"],
        name="max_div",
        title="Maximum # of Divergences",
        width=200,
    )

    convergence_threshold = build_model(
        TextInput,
        toml_control_path_list=["convergence_threshold"],
        name="convergence_threshold",
        title="Convergence Threshold",
        width=200,
    )

    initial_gamma_lm = build_model(
        TextInput,
        toml_control_path_list=["initial_gamma_lm"],
        name="initial_gamma_lm",
        title="Initial Levenberg-Marquardt Parameter",
        width=200,
    )

    r_divergent = build_model(
        TextInput,
        toml_control_path_list=["r_divergent"],
        name="r_divergent",
        title="Divergent Ratio",
        width=200,
    )

    r_convergent = build_model(
        TextInput,
        toml_control_path_list=["r_convergent"],
        name="r_convergent",
        title="Convergent Ratio",
        width=200,
    )

    divergent_scale_factor = build_model(
        TextInput,
        toml_control_path_list=["divergent_scale_factor"],
        name="divergent_scale_factor",
        title="Divergent Scale Factor",
        width=200,
    )

    convergent_scale_factor = build_model(
        TextInput,
        toml_control_path_list=["convergent_scale_factor"],
        name="convergent_scale_factor",
        title="Convergent Scale Factor",
        width=200,
    )

    chi_gof_threshold = build_model(
        TextInput,
        toml_control_path_list=["chi_gof_threshold"],
        name="chi_gof_threshold",
        title="Chi-squared Threshold",
        width=200,
    )

    scale_obs_cov = build_model(
        Select,
        toml_control_path_list=["scale_obs_cov"],
        is_bool=True,
        linked_model_list=["scale_obs_cov_inputs"],
        title="Scale Observation Covariance",
        options=["T", "F"],
        name="scale_obs_cov",
        width=200,
    )

    scale_obs_cov_inputs = build_model(
        TextInput,
        toml_control_path_list=["obs_cov_scale_factor"],
        name="scale_obs_cov_inputs",
        title="Scale Factor",
        width=200,
    )

    scale_prior_cov = build_model(
        Select,
        toml_control_path_list=["scale_prior_cov"],
        is_bool=True,
        linked_model_list=["scale_prior_cov_inputs"],
        title="Scale A Priori Covariance",
        options=["T", "F"],
        name="scale_prior_cov",
        width=200,
    )

    scale_prior_cov_inputs = build_model(
        TextInput,
        toml_control_path_list=["prior_cov_scale_factor"],
        name="scale_prior_cov_inputs",
        title="Scale Factor",
        width=200,
    )

    init_state_with_prev_fit = build_model(
        Select,
        toml_control_path_list=["init_state.init_state_with_prev_fit"],
        is_bool=True,
        linked_model_list=["init_state_with_prev_fit_inputs"],
        title="Initialize State with Previous Fit",
        options=["T", "F"],
        name="init_state_with_prev_fit",
        width=200,
    )

    filter_chi_squared = build_model(
        Select,
        toml_control_path_list=["init_state.filter_chi_squared"],
        is_bool=True,
        linked_model_list=["max_chi_squared"],
        title="Filter Chi-squared",
        options=["T", "F"],
        name="filter_chi_squared",
        width=200,
    )

    max_chi_squared = build_model(
        TextInput,
        toml_control_path_list=["init_state.max_chi_squared"],
        name="max_chi_squared",
        title="Maximum Chi-squared",
        width=200,
        visible=control_data["init_state"]["filter_chi_squared"] == "T",
    )

    filter_rms = build_model(
        Select,
        toml_control_path_list=["init_state.filter_rms"],
        is_bool=True,
        linked_model_list=["max_rms"],
        title="Filter RMS",
        options=["T", "F"],
        name="filter_rms",
        width=200,
    )

    max_rms = build_model(
        TextInput,
        toml_control_path_list=["init_state.max_rms"],
        name="max_rms",
        title="Maximum RMS",
        width=200,
        visible=control_data["init_state"]["filter_rms"] == "T",
    )

    substate_list = [
        f"{i}_substate"
        for i in [
            "gas",
            "aerosol",
            "temperature",
            "surface_pressure",
            "brdf",
            "radiance_offset",
            "radiance_scaling",
            "residual_eof",
            "isrf",
            "wavelength",
        ]
    ]

    substate_model_list = [
        build_model(
            Select,
            toml_control_path_list=[f"init_state.{model_name}"],
            is_bool=True,
            title=" ".join(model_name.split("_")),
            options=["T", "F"],
            name=model_name,
            width=200,
        )
        for model_name in substate_list
    ]

    init_state_with_prev_fit_inputs = column(
        children=[filter_chi_squared, max_chi_squared, filter_rms, max_rms] + substate_model_list,
        name="init_state_with_prev_fit_inputs",
        visible=control_data["init_state"]["init_state_with_prev_fit"] == "T",
    )

    optimizer_inputs = column(
        children=[
            max_iter,
            max_div,
            convergence_threshold,
            initial_gamma_lm,
            row(children=[r_divergent, r_convergent]),
            row(children=[divergent_scale_factor, convergent_scale_factor]),
            chi_gof_threshold,
            scale_obs_cov,
            scale_obs_cov_inputs,
            scale_prior_cov,
            scale_prior_cov_inputs,
            init_state_with_prev_fit,
            init_state_with_prev_fit_inputs,
        ],
        name="optimizer_inputs",
        width=1600,
    )

    return TabPanel(title="Optimizer", child=optimizer_inputs, name="optimizer_panel")


def diagnostics_options() -> bokeh.models.layouts.TabPanel:
    """
    Generate the layouts in the Diagnostics panel
    """
    viewing_geometry = build_model(
        Select,
        toml_control_path_list=["level1_options.viewing_geometry"],
        is_bool=True,
        options=["T", "F"],
        name="viewing_geometry",
        title="Viewing Geometry",
        width=200,
    )

    level1_diag_options = TabPanel(
        title="Level1 Diag", child=column(children=[viewing_geometry]), name="level1_diag"
    )

    profile_output_dict = {
        "profile_output_options.air_molecular_weight": {
            "title": "Air Molecular Weight",
            "species_builder": None,
        },
        "profile_output_options.gravity": {"title": "Gravity", "species_builder": None},
        "profile_output_options.relative_humidity": {
            "title": "Relative Humidity",
            "species_builder": None,
        },
        "profile_output_options.air_partial_column": {
            "title": "Air Partial Column",
            "species_builder": None,
        },
        "profile_output_options.total_aod": {"title": "Total AOD", "species_builder": None},
        "profile_output_options.cloud_n_pixels": {
            "title": "# of Cloud Pixels",
            "species_builder": None,
        },
        "profile_output_options.cloud_fraction_total": {
            "title": "Total Cloud Fraction",
            "species_builder": None,
        },
        "profile_output_options.total_cod": {
            "title": "Total Cloud Optical Depth",
            "species_builder": None,
        },
        "profile_output_options.pressure.layer_edge": {
            "title": "Pressure (Layer Edge)",
            "species_builder": None,
        },
        "profile_output_options.pressure.layer_mid": {
            "title": "Pressure (Layer Mid)",
            "species_builder": None,
        },
        "profile_output_options.temperature.layer_edge": {
            "title": "Temperature (Layer Edge)",
            "species_builder": None,
        },
        "profile_output_options.temperature.layer_mid": {
            "title": "Temperature (Layer Mid)",
            "species_builder": None,
        },
        "profile_output_options.altitude.layer_edge": {
            "title": "Altitude (Layer Edge)",
            "species_builder": None,
        },
        "profile_output_options.altitude.layer_mid": {
            "title": "Altitude (Layer Mid)",
            "species_builder": None,
        },
        "profile_output_options.gas_partial_column": {
            "title": "Gas Partial Column",
            "species_builder": gas_checkboxes,
        },
        "profile_output_options.gas_column_uncertainties": {
            "title": "Gas Column Uncertainty",
            "species_builder": gas_checkboxes,
        },
        "profile_output_options.gas_mixing_ratios": {
            "title": "Gas Mixing Ratio",
            "species_builder": gas_checkboxes,
        },
        "profile_output_options.dry_gas_mixing_ratios": {
            "title": "Gas Dry-Air Mixing Ratio",
            "species_builder": gas_checkboxes,
        },
        "profile_output_options.proxy_column_avg_mix_rat": {
            "title": "Proxy Column-Averaged Mixing Ratio",
            "species_builder": gas_checkboxes,
        },
        "profile_output_options.species_aod": {
            "title": "Species AOD",
            "species_builder": aerosol_checkboxes,
        },
        "profile_output_options.species_profile_params": {
            "title": "Species Profile Parameters",
            "species_builder": aerosol_checkboxes,
        },
        "profile_output_options.species_profile_par_derivs": {
            "title": "Species Profile Par. Deriv.",
            "species_builder": aerosol_checkboxes,
        },
    }

    profile_output_column = column(
        children=[
            build_diag_model(
                title=v["title"], toml_control_path=k, species_builder=v["species_builder"]
            )
            for k, v in profile_output_dict.items()
        ],
        name="profile_output_column",
    )

    profile_output_options = TabPanel(
        title="Profile Diag", child=profile_output_column, name="profile_diag"
    )

    rtm_output_dict = {
        "rtm_output_options.wavelength": {"title": "Wavelength", "species_builder": None},
        "rtm_output_options.irradiance": {"title": "Irradiance", "species_builder": None},
        "rtm_output_options.radiance": {"title": "Radiance", "species_builder": None},
        "rtm_output_options.radiant_flux": {"title": "Radiant Flux", "species_builder": None},
        "rtm_output_options.direct_flux": {"title": "Direct Flux", "species_builder": None},
    }

    rtm_output_column = column(
        children=[
            build_diag_model(
                title=v["title"], toml_control_path=k, species_builder=v["species_builder"]
            )
            for k, v in rtm_output_dict.items()
        ],
        name="rtm_output_column",
    )

    rtm_output_options = TabPanel(title="RTM Diag.", child=rtm_output_column, name="rtm_diag")

    rtm_output_gas_jacobians_dict = {
        "rtm_output_options.gas_jacobians.scattering_weights": {
            "title": "Scattering Weights",
            "species_builder": None,
        },
        "rtm_output_options.gas_jacobians.trace_gas": {
            "title": "Trace Gas",
            "species_builder": None,
        },
        "rtm_output_options.gas_jacobians.trace_gas_part_col": {
            "title": "Trace Gas Partial Column",
            "species_builder": None,
        },
        "rtm_output_options.gas_jacobians.air_mass_factors": {
            "title": "Airmass Factors",
            "species_builder": None,
        },
    }

    rtm_output_gas_jacobians_column = column(
        children=[
            build_diag_model(
                title=v["title"], toml_control_path=k, species_builder=v["species_builder"]
            )
            for k, v in rtm_output_gas_jacobians_dict.items()
        ],
        name="rtm_output_gas_jacobians_column",
    )

    rtm_output_gas_jacobians_options = TabPanel(
        title="RTM Gas Jac.", child=rtm_output_gas_jacobians_column, name="rtm_gas_jacobians_diag"
    )

    rtm_output_met_jacobians_dict = {
        "rtm_output_options.met_jacobians.temperature_profile": {
            "title": "Temperature Profile",
            "species_builder": None,
        },
        "rtm_output_options.met_jacobians.temperature_shift": {
            "title": "Temperature Shift",
            "species_builder": None,
        },
        "rtm_output_options.met_jacobians.surface_pressure": {
            "title": "Surface Pressure",
            "species_builder": None,
        },
    }

    rtm_output_met_jacobians_column = column(
        children=[
            build_diag_model(
                title=v["title"], toml_control_path=k, species_builder=v["species_builder"]
            )
            for k, v in rtm_output_met_jacobians_dict.items()
        ],
        name="rtm_output_met_jacobians_column",
    )

    rtm_output_met_jacobians_options = TabPanel(
        title="RTM Met Jac.", child=rtm_output_met_jacobians_column, name="rtm_met_jacobians_diag"
    )

    rtm_output_aerosol_jacobians_dict = {
        "rtm_output_options.aerosol_jacobians.profile_aod": {
            "title": "Profile AOD",
            "species_builder": aerosol_checkboxes,
        },
        "rtm_output_options.aerosol_jacobians.profile_aod_par": {
            "title": "Profile AOD Parameters",
            "species_builder": aerosol_checkboxes,
        },
        "rtm_output_options.aerosol_jacobians.profile_ssa": {
            "title": "Profile SSA",
            "species_builder": aerosol_checkboxes,
        },
        "rtm_output_options.aerosol_jacobians.profile_ssa_par": {
            "title": "Profile SSA Parameters",
            "species_builder": aerosol_checkboxes,
        },
        "rtm_output_options.aerosol_jacobians.opt_prop_params": {
            "title": "Optical Properties Parameters",
            "species_builder": aerosol_checkboxes,
        },
    }

    rtm_output_aerosol_jacobians_column = column(
        children=[
            build_diag_model(
                title=v["title"], toml_control_path=k, species_builder=v["species_builder"]
            )
            for k, v in rtm_output_aerosol_jacobians_dict.items()
        ],
        name="rtm_output_aerosol_jacobians_column",
    )

    rtm_output_aerosol_jacobians_options = TabPanel(
        title="RTM Aerosol Jac.",
        child=rtm_output_aerosol_jacobians_column,
        name="rtm_aerosol_jacobians_diag",
    )

    rtm_output_surface_jacobians_dict = {
        "rtm_output_options.surface_jacobians.brdf_kernel_factors": {
            "title": "BRDF Kernel Factors",
            "species_builder": None,
        },
        "rtm_output_options.surface_jacobians.brdf_kernel_parameters": {
            "title": "BRDF Kernel Parameters",
            "species_builder": None,
        },
    }

    rtm_output_surface_jacobians_column = column(
        children=[
            build_diag_model(
                title=v["title"], toml_control_path=k, species_builder=v["species_builder"]
            )
            for k, v in rtm_output_surface_jacobians_dict.items()
        ],
        name="rtm_output_surface_jacobians_column",
    )

    rtm_output_surface_jacobians_options = TabPanel(
        title="RTM Surface Jac.",
        child=rtm_output_surface_jacobians_column,
        name="rtm_surface_jacobians_diag",
    )

    rtm_output_instrument_jacobians_dict = {
        "rtm_output_options.instrument_jacobians.isrf": {"title": "ISRF", "species_builder": None},
        "rtm_output_options.instrument_jacobians.wavelength_shift": {
            "title": "Wavelength Shift",
            "species_builder": None,
        },
    }

    rtm_output_instrument_jacobians_column = column(
        children=[
            build_diag_model(
                title=v["title"], toml_control_path=k, species_builder=v["species_builder"]
            )
            for k, v in rtm_output_instrument_jacobians_dict.items()
        ],
        name="rtm_output_instrument_jacobians_column",
    )

    rtm_output_instrument_jacobians_options = TabPanel(
        title="RTM Instrument Jac.",
        child=rtm_output_instrument_jacobians_column,
        name="rtm_instrument_jacobians_diag",
    )

    optical_property_diagnostics_dict = {
        "optical_property_diagnostics.total_optical_depth": {
            "title": "Total Optical Depth",
            "species_builder": None,
        },
        "optical_property_diagnostics.brdf_kernel_amplitudes": {
            "title": "BRDF Kernel Amplitudes",
            "species_builder": None,
        },
        "optical_property_diagnostics.brdf_kernel_amplitude_ref_wvl": {
            "title": "BRDF Kernel Amplitude at Reference Wavelength",
            "species_builder": None,
        },
        "optical_property_diagnostics.gas_absorption_xsect": {
            "title": "Gas Absorption Cross-Section",
            "species_builder": gas_checkboxes,
        },
    }

    optical_property_diagnostics_column = column(
        children=[
            build_diag_model(
                title=v["title"], toml_control_path=k, species_builder=v["species_builder"]
            )
            for k, v in optical_property_diagnostics_dict.items()
        ],
        name="optical_property_diagnostics_column",
    )

    optical_property_diagnostics = TabPanel(
        title="Optical Prop. Diag.", child=optical_property_diagnostics_column, name="opt_prop_diag"
    )

    diagnostics_tabs = Tabs(
        tabs=[
            level1_diag_options,
            profile_output_options,
            rtm_output_options,
            rtm_output_gas_jacobians_options,
            rtm_output_met_jacobians_options,
            rtm_output_aerosol_jacobians_options,
            rtm_output_surface_jacobians_options,
            rtm_output_instrument_jacobians_options,
            optical_property_diagnostics,
        ],
        name="diagnostics_tabs",
        stylesheets=[
            InlineStyleSheet(
                css="""
                    div.bk-tab {
                        background-color: mistyrose;
                        font-weight: bold;
                        border-color: darkgray;
                        color: mediumpurple;
                    }
                    div.bk-tab.bk-active {
                        background-color: lightpink;
                        border-color: mediumpurple;
                        color: mediumpurple;
                        font-weight: bold;
                    }
                    div.bk-header {
                        border-bottom: 0px !important;
                    }
                    """
            )
        ],
    )

    return TabPanel(title="Diagnostics", child=diagnostics_tabs, name="diagnostics_panel")


def inverse_diagnostics_options() -> bokeh.models.layouts.TabPanel:
    """
    Generate the layouts in the Inverse Diagnostics panel
    """

    general_fit_statistics_dict = {
        "general_fit_statistics.cost_function": {"title": "Cost Function"},
        "general_fit_statistics.chi_square": {"title": "Chi-squared"},
        "general_fit_statistics.uncertainty_derivative": {"title": "Uncertainty Derivative"},
        "general_fit_statistics.spectrum_rms": {"title": "Spectrum RMS"},
        "general_fit_statistics.number_of_iterations": {"title": "Number of Iterations"},
        "general_fit_statistics.fit_quality_flag": {"title": "Fit Quality Flag"},
    }

    general_fit_statistics_selectors = []
    for k, v in general_fit_statistics_dict.items():
        general_fit_statistics_selectors += [
            build_model(
                Select,
                toml_control_path_list=[k],
                name=k.replace(".", "_"),
                title=v["title"],
                is_bool=True,
                options=["T", "F"],
                width=200,
            )
        ]

    general_fit_statistics_options = TabPanel(
        title="General Fit Stats",
        child=column(children=general_fit_statistics_selectors),
        name="general_fit_statistics_options",
    )

    spectrum_dict = {
        "spectrum.spectrum_residuals": {"title": "Spectrum Residuals"},
        "spectrum.level1_spectrum": {"title": "Level1 Spectrum"},
        "spectrum.level1_uncertainty": {"title": "Level1 Uncertainty"},
    }

    spectrum_selectors = []
    for k, v in spectrum_dict.items():
        spectrum_selectors += [
            build_model(
                Select,
                toml_control_path_list=[k],
                name=k.replace(".", "_"),
                title=v["title"],
                is_bool=True,
                options=["T", "F"],
                width=200,
            )
        ]
    iteration_radiance = build_model(
        Select,
        toml_control_path_list=["spectrum.iteration_radiance"],
        linked_model_list=["iteration_radiance_inputs"],
        name="iteration_radiance",
        title="Iteration Radiance",
        is_bool=True,
        options=["T", "F"],
        width=200,
    )
    iteration_radiance_inputs = build_model(
        TextInput,
        toml_control_path_list=["spectrum.iteration_radiance_max_iter"],
        name="iteration_radiance_inputs",
        title="Maximum # of Iterations",
        width=200,
        visible=control_data["spectrum"]["iteration_radiance"] == "T",
    )

    spectrum_options = TabPanel(
        title="Spectrum",
        child=column(children=spectrum_selectors + [iteration_radiance, iteration_radiance_inputs]),
        name="spectrum_options",
    )

    total_state_vector_dict = {
        "total_state_vector.averaging_kernel": {"title": "Averaging Kernel"},
        "total_state_vector.posteriori_covariance": {"title": "Posterior Covariance"},
        "total_state_vector.priori_covariance": {"title": "Prior Covariance"},
        "total_state_vector.posteriori_state": {"title": "Posterior State"},
        "total_state_vector.priori_state": {"title": "Prior State"},
    }

    total_state_vector_selectors = []
    for k, v in total_state_vector_dict.items():
        total_state_vector_selectors += [
            build_model(
                Select,
                toml_control_path_list=[k],
                name=k.replace(".", "_"),
                title=v["title"],
                is_bool=True,
                options=["T", "F"],
                width=200,
            )
        ]
    iteration_state = build_model(
        Select,
        toml_control_path_list=["total_state_vector.iteration_state"],
        linked_model_list=["iteration_state_inputs"],
        name="iteration_state",
        title="Iteration State",
        is_bool=True,
        options=["T", "F"],
        width=200,
    )
    iteration_state_inputs = build_model(
        TextInput,
        toml_control_path_list=["total_state_vector.iteration_state_max_iter"],
        name="iteration_state_inputs",
        title="Maximum # of Iterations",
        width=200,
        visible=control_data["total_state_vector"]["iteration_state"] == "T",
    )

    total_state_vector_options = TabPanel(
        title="Total State Vector",
        child=column(
            children=total_state_vector_selectors + [iteration_state, iteration_state_inputs]
        ),
        name="total_state_vector_options",
    )

    sub_state_title_div = custom_div(text="Gas Absorption Posterior Covariance")
    sub_state_vector_dict = {
        "sub_state_vector_diagnostics.gas_absorption_post_covar.profile_mixing_ratio": {
            "title": "Profile Mixing Ratio",
            "species_builder": gas_checkboxes,
        },
        "sub_state_vector_diagnostics.gas_absorption_post_covar.column_avg_mixing_ratio": {
            "title": "Column-Averaged Mixing Ratio",
            "species_builder": gas_checkboxes,
        },
        "sub_state_vector_diagnostics.gas_absorption_post_covar.dry_profile_mixing_ratio": {
            "title": "Profile Dry-Air Mixing Ratio",
            "species_builder": gas_checkboxes,
        },
        "sub_state_vector_diagnostics.gas_absorption_post_covar.dry_column_avg_mixing_ratio": {
            "title": "Column-Averaged Dry-Air Mixing Ratio",
            "species_builder": gas_checkboxes,
        },
        "sub_state_vector_diagnostics.gas_absorption_post_covar.proxy_posteriori_uncert": {
            "title": "Proxy Posterior Uncertainty",
            "species_builder": gas_checkboxes,
        },
        "sub_state_vector_diagnostics.gas_absorption_post_covar.column_averaging_kernel": {
            "title": "Column Averaging Kernel",
            "species_builder": gas_checkboxes,
        },
    }

    sub_state_vector_models = [
        build_inv_diag_model(
            title=v["title"], toml_control_path=k, species_builder=v["species_builder"]
        )
        for k, v in sub_state_vector_dict.items()
    ]

    sub_state_vector_options = TabPanel(
        title="Sub State Vector",
        child=column(children=[sub_state_title_div] + sub_state_vector_models),
        name="sub_state_vector_options",
    )

    gain_matrix = build_model(
        Select,
        toml_control_path_list=["full_matrices_for_offline_err.gain_matrix"],
        name="gain_matrix",
        title="Gain Matrix",
        is_bool=True,
        options=["T", "F"],
        width=200,
    )

    full_matrices_options = TabPanel(
        title="Full Matrices", child=column(children=[gain_matrix]), name="full_matrices_options"
    )

    # error analysis
    error_analysis_models = [
        gas_checkboxes(
            name=i,
            toml_control_path=f"error_analysis.{i}",
            # fmt: off
            default_active=control_data["error_analysis"][i] if i!="print_outputs" else [],
            # fmt: on
            inline=True,
            as_bool=i == "print_outputs",
            linked_outputs="gas_column_components_checkboxes",
        )
        for i in ["proxy_name", "gas_column_components", "print_outputs", "proxy_xgas_components"]
    ]

    error_analysis_inputs = column(
        children=error_analysis_models,
        name="error_analysis_inputs",
        visible=control_data["error_analysis"]["do_error_analysis"] == "T",
    )

    do_error_analysis = build_model(
        Select,
        toml_control_path_list=["error_analysis.do_error_analysis"],
        name="do_error_analysis",
        title="Do Error Analysis",
        linked_model_list=["error_analysis_inputs"],
        is_bool=True,
        options=["T", "F"],
        width=200,
        stylesheets=select_status_dict[control_data["error_analysis"]["do_error_analysis"]],
    )

    error_analysis_options = TabPanel(
        title="Error Analysis",
        child=column(children=[do_error_analysis, error_analysis_inputs]),
        name="error_analysis_options",
    )

    inverse_diagnostics_tabs = Tabs(
        tabs=[
            general_fit_statistics_options,
            spectrum_options,
            total_state_vector_options,
            sub_state_vector_options,
            full_matrices_options,
            error_analysis_options,
        ],
        name="inverse_diagnostics_tabs",
        stylesheets=[
            InlineStyleSheet(
                css="""
                    div.bk-tab {
                        background-color: mistyrose;
                        font-weight: bold;
                        border-color: darkgray;
                        color: mediumpurple;
                    }
                    div.bk-tab.bk-active {
                        background-color: lightpink;
                        border-color: mediumpurple;
                        color: mediumpurple;
                        font-weight: bold;
                    }
                    div.bk-header {
                        border-bottom: 0px !important;
                    }
                    """
            )
        ],
    )

    return TabPanel(
        title="Inverse Diagnostics",
        child=inverse_diagnostics_tabs,
        name="inverse_diagnostics_panel",
    )


def doc_maker():
    """
    Function that builds the elements in the document
    """
    global control_data

    # remove everything in the current document
    curdoc().clear()

    # build the different panels and gather them in a Tabs object
    control = Tabs(
        tabs=[
            globals()[f"{i}_options"]()
            for i in [
                "file",
                "retrieval",
                "level1",
                "rtm",
                "window",
                "profile",
                "surface",
                "gas",
                "aerosol",
                "cloud",
                "state_vector",
                "optimizer",
                "diagnostics",
                "inverse_diagnostics",
            ]
        ],
        stylesheets=[
            InlineStyleSheet(
                css="""
                    div.bk-tab {
                        background-color: lightcyan;
                        font-weight: bold;
                        border-color: darkgray;
                        color: teal;
                    }
                    div.bk-tab.bk-active {
                        background-color: lightblue;
                        border-color: teal;
                        color: teal;
                        font-weight: bold;
                    }
                    div.bk-header {
                        border-bottom: 0px !important;
                    }
                    """
            )
        ],
    )

    control_output = TextInput(
        name="control_output",
        title="Full path to the output SPLAT control file (including filename WITHOUT extension)",
    )
    save_button = Button(
        label="Save",
        width=80,
        stylesheets=[
            InlineStyleSheet(
                css="""
                    div.bk-btn-group button {
                        border:none;
                        background-color: lightblue;
                        color:teal;
                        font-weight: bold;
                        height: 40px !important;
                    }
                    """
            )
        ],
    )
    save_button.on_click(save_control_file)

    save_options = row(children=[control_output, save_button])

    control_input = TextInput(
        name="control_input",
        title="Full path to the input SPLAT control file (.TOML format)",
    )
    load_button = Button(
        label="Load",
        width=80,
        stylesheets=[
            InlineStyleSheet(
                css="""
                    div.bk-btn-group button {
                        border:none;
                        background-color: lightblue;
                        color:teal;
                        font-weight: bold;
                        height: 40px !important;
                    }
                    """
            )
        ],
    )
    load_button.on_click(load_toml)

    # Dummy invisible widget used to reload the page at the end of load_toml()
    reload_page_input = TextInput(
        value="",
        name="reload_page_input",
        visible=False,
    )
    reload_page_input.js_on_change("value", CustomJS(code="window.location.reload();"))

    load_options = row(children=[control_input, load_button, reload_page_input])

    final_layout = column(children=[load_options, save_options, control])

    curdoc().add_root(final_layout)


def modify_doc(doc):
    """
    Initialize the Bokeh document
    """
    global select_status_dict

    curstate().document = doc

    # this is displayed in the browser tab
    curdoc().title = "SPLAT"

    # define it here as a global variable to be accessible from callbacks and to allow reloading the document without model conflicts
    select_status_dict = {
        "T": [
            InlineStyleSheet(
                css="div.bk-input-group select.bk-input {background-color: lightgreen}"
            )
        ],
        "F": [
            InlineStyleSheet(
                css="div.bk-input-group select.bk-input {background-color: darksalmon}"
            )
        ],
    }

    # fill the document
    doc_maker()


def start_server(toml_file: str, jinja_template_file: str):
    """
    Launch the bokeh server and connect to it.

    toml_file (str): full path to the input .toml control file
    """
    global control_data, template_file

    template_file = jinja_template_file

    # Having the data available globally from here
    # Then it does not get reloaded when we refresh the page
    with open(toml_file, "r") as infile:
        control_data = toml.load(infile)

    io_loop = IOLoop.current()
    bokeh_app = Application(FunctionHandler(modify_doc))

    server = Server({"/": bokeh_app}, io_loop=io_loop)
    server.start()

    io_loop.add_callback(server.show, "/")
    io_loop.start()


def main():
    """
    Driver function to launch the bokeh app
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--toml-file",
        default=os.path.join(app_path, "inputs", "tomltest_o2.toml"),
        help="Full path to the input TOML file",
    )
    parser.add_argument(
        "-j",
        "--jinja-template",
        default=os.path.join(app_path, "inputs", "template.control"),
        help="Full path to the jinja template .control file",
    )
    args = parser.parse_args()

    start_server(args.toml_file, args.jinja_template)


if __name__ == "__main__":
    main()
