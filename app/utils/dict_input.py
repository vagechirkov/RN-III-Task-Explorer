# Copied from https://github.com/robwalton/streamlit-toy-dict-input
# Thanks Rob Walton!

import json
from dataclasses import dataclass
from typing import Dict
import inspect

import streamlit as st


PIXELS_PER_LINE = 5
INDENT = 8


@st.cache(allow_output_mutation=True)
def state_singleton() -> Dict:
    return {}


STATE = state_singleton()


@dataclass
class JsonInputState:
    value: dict
    default_value: dict
    redraw_counter = 0


def dict_input(label, value, mutable_structure=False, key=None):
    """Display a dictionary or dictionary input widget.

    This implementation is composed of a number of streamlit widgets. It might
    be considered a prototype for a native streamlit widget (perhaps built off
    the existing interactive dictionary widget).

    Json text may be copied in and out of the widget.

    Parameters
    ----------
    label : str
        A short label explaining to the user what this input is for.
    value : dict or func
        The dictionary of values to edit or a function (with only named parameters).
    mutable_structure : bool
        If True allows changes to the structure of the initial value.
        Otherwise the keys and the type of their values are fixed.
        Defaults to False (non mutable).
    key : str
        An optional string to use as the unique key for the widget.
        If this is omitted, a key will be generated for the widget
        based on its content. Multiple widgets of the same type may
        not share the same key.


    Returns
    -------
    dict
        The current value of the input widget.

    """
    try:
        param = inspect.signature(value).parameters
        value = {}
        for p in param.values():
            value[p.name] = p.default
    except TypeError:
        pass  # Assume value is a dict

    # check json can handle input
    value = json.loads(json.dumps(value))

    # Create state on first run
    state_key = f"json_input-{key if key else label}"
    if state_key not in STATE:
        STATE[state_key] = JsonInputState(value, value)
    state: JsonInputState = STATE[state_key]

    # containers
    text_con = st.empty()
    warning_con = st.empty()

    def json_input_text(msg=""):

        if msg:
            state.redraw_counter += 1
            state.default_value = state.value

        # Display warning
        if msg:
            warning_con.warning(msg)
        else:
            warning_con.empty()

        # Read value
        value_s = json.dumps(
            state.default_value, indent=INDENT, sort_keys=True
        )
        input_s = text_con.text_area(
            label,
            value_s,
            height=len(value_s.splitlines()) * PIXELS_PER_LINE,
            key=f"{key if key else label}-{state.redraw_counter}",
            # help="help"
        )

        # Decode
        try:
            new_value = json.loads(input_s)
        except json.decoder.JSONDecodeError:
            return json_input_text(
                "The last edit was invalid json and has been reverted"
            )

        # Check structure
        if not mutable_structure:
            if not keys_match(new_value, state.value):
                return json_input_text(
                    "The last edit changed the structure of the json "
                    "and has been reverted"
                )

            if not value_types_match(new_value, state.value):
                return json_input_text(
                    "The last edit changed the type of an entry "
                    "and has been reverted"
                )

        return new_value

    # Input a valid dict
    state.value = json_input_text()

    return state.value


def keys_match(d1, d2):
    if d1.keys() != d2.keys():
        return False

    for k, v in d1.items():
        if isinstance(v, dict):
            if not keys_match(v, d2[k]):
                return False

    for k, v in d2.items():
        if isinstance(v, dict):
            if not keys_match(v, d1[k]):
                return False

    return True


def value_types_match(d1, d2):
    # assuming the keys match
    for k in d1.keys():
        if isinstance(d1[k], dict):
            if not value_types_match(d1[k], d2[k]):
                return False
        if type(d1[k]) is not type(d2[k]):
            return False

    return True
