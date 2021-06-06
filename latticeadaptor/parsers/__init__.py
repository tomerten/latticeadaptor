# -*- coding: utf-8 -*-

"""
Module latticeadaptor.parsers 
=================================================================

A module containg accelerator lattice parsers for converting between formats.

"""

# your imports here ...
from abc import ABC
from json import load
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from lark import Lark, Transformer, v_args
from lark.exceptions import LarkError

# ==============================================================================
# SET BASE DIR
# LOAD MAPPINGS BETWEEN THE DIFFERENT FORMATS
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent

with (BASE_DIR / "../lark/Madx.lark").open() as file:
    MADX_PARSER = Lark(file, parser="lalr", maybe_placeholders=True)
    file.seek(0)

# MADX
with (BASE_DIR / "../mappings/madx_columns.json").open() as file:
    MADX_ATTRIBUTES = load(file)

# ELEGANT
with (BASE_DIR / "../mappings/elegant_columns.json").open() as file:
    ELEGANT_ATTRIBUTES = load(file)

with (BASE_DIR / "../mappings/elegant_element_map.json").open() as file:
    TO_ELEGANT_ELEMENTS = load(file)

with (BASE_DIR / "../mappings/elegant_attribute_map.json").open() as file:
    TO_ELEGANT_ATTR = load(file)

# TRACY
with (BASE_DIR / "../mappings/tracy_columns.json").open() as file:
    TRACY_ATTRIBUTES = load(file)

with (BASE_DIR / "../mappings/tracy_element_map.json").open() as file:
    TO_TRACY_ELEMENTS = load(file)

with (BASE_DIR / "../mappings/tracy_attribute_map.json").open() as file:
    TO_TRACY_ATTR = load(file)


def save_string(string: str, file: str) -> None:
    """Quick method to save string to file.add()

    Parameters
    ----------
    string : str
        String to be saved to file.add()
    file : str
        filename where to save the string
    """
    with open(file, "w") as f:
        f.write(string)


@v_args(inline=True)
class AbstractSequenceFileTransformer(ABC, Transformer):
    """Lark Transformer Class"""

    def transform(self, tree):
        """Method to transform the parsed data"""
        self.elements = []
        self.seq = None
        self.name = None
        self.length = 0.0
        super().transform(tree)
        return self.seq, self.elements, self.name, self.length

    int = int
    float = float
    word = str
    neg = lambda self, item: -item
    number = float
    name = lambda self, item: item.value.upper()
    string = lambda self, item: item[1:-1]

    def element(self, name, type_, *attributes):
        # self.elements[name.upper()] = {**{'family': type_.upper()}, **dict(attributes)}
        self.elements.append(
            {**{"name": name.upper(), "family": type_.upper()}, **dict(attributes)}
        )
        return name

    def attribute(self, name, value):
        return name.upper(), value

    def seq_element(self, name, value):
        return name.upper(), value

    def sequence(self, name, *attr):
        self.name = name
        self.length = attr[0][1]
        return name, attr

    def seq_elements(self, *attr):
        self.seq = attr

    def true(self, *attr):
        return True


@v_args(inline=True)
class MADXTransformer(AbstractSequenceFileTransformer):
    pass


from typing import Tuple, Union

from latticeconstructor.parse import parse_elegant, parse_madx
from latticejson.convert import FROM_ELEGANT, TO_MADX

_CONVERSION_DICT = {
    "KQUAD": "QUADRUPOLE",
    "KSEXT": "SEXTUPOLE",
    "KOCT": "OCTUPOLE",
    "DRIF": "DRIFT",
    "RFCA": "RFCAVITY",
    "CSBEND": "SBEND",
    "MONI": "MONITOR",
    "WATCH": "MARKER",
    "EVKICK": "VKICKER",
    "EHKICK": "HKICKER",
    "MARK": "MARKER",
}


def parse_from_string(string: str, ftype: str = "lte") -> Tuple[Union[str, None], dict, list]:
    """Method to parse an elegant lattice string to name, defintions and ordered
    lattice element list.

    Parameters
    ----------
    string : str
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    Lattice string.

    ftype: optional, str, default = 'lte'
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    format type

    Returns
    -------
    Tuple[Union[str,None],dict, list]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    name, element definitions dict, element ordered list

    """
    assert ftype.lower() in ["lte", "madx"]

    # use latticejson parser
    if ftype == "lte":
        latdata = parse_elegant(string)
    else:
        latdata = parse_madx(string)

    # create command dict for later use
    cdict = {}
    for command in latdata.get("commands"):
        try:
            cdict[command[0]] = command[1] if len(command[1]) == 1 else list(command[1])
        except Exception:
            cdict[command[0]] = ""

    # get the element definitions dict
    definitions = latdata.get("elements", [])

    # convert to madx names
    definitions = {}
    for name, (_type, attributes) in latdata.get("elements", []).items():
        madtype = _type.upper()
        if ftype == "lte":
            # if no matching type make a marker
            try:
                madtype = TO_MADX[FROM_ELEGANT[_type.lower()]].upper()
            except Exception:
                madtype = _CONVERSION_DICT[_type.upper()]

        definitions[name.upper()] = {
            **{"family": madtype},
            **{k.upper(): v for k, v in attributes.items()},
        }

    # get the lattice sub-lattice dicts
    lattice = latdata.get("lattices", [])

    # method to flatten the sublattices
    def flatten(sublat_dict):
        def _walker(k, lattices=sublat_dict):
            for child in lattices[k]:
                if child in lattices:
                    yield from _walker(child)
                else:
                    yield child

        return _walker(list(sublat_dict.keys())[-1])

    # load ordered list to lattice
    # TODO: requested madx sequence lark parser to be merged in main branch so the
    # hack below can be avoided
    if bool(lattice):
        lattice = list(flatten(lattice))
        lattice = [el.upper() for el in lattice]
    else:
        cdict.pop("ENDSEQUENCE")
        lattice = [el.upper() for el in list(cdict.keys())]

    # attempt to load lattice name
    lattice_name = cdict.get("use", None)

    return lattice_name, definitions, lattice


def parse_from_madx_sequence_string(string: str) -> Tuple[str, float, pd.DataFrame]:
    """Method to parse madx seq string to table format

    :param str string: Madx sequence string
    :returns: name, length, dataframe containing the lattice elements
    """
    # use lark to parse the string
    tree = MADX_PARSER.parse(string)
    positions, elements, name, length = MADXTransformer().transform(tree)

    # read the positions of the elements ('at' in the seq file)
    if positions is not None:
        dfpos = pd.DataFrame.from_records(positions, columns=["name", "pos"])
    else:
        dfpos = pd.DataFrame()

    # if not bare sequence file
    if elements:
        dfel = pd.DataFrame(elements)

        # if positions are available merge the tables
        if positions:
            df = dfpos.merge(dfel, on="name").sort_values(by="pos")
            df.loc[df.L.isna(), "L"] = 0
            df["at"] = df["pos"]
            return name, length, df
        else:
            return name, length, dfel

    # if seq file is bare print warning and return only
    # pos table as table
    print("Warning: bare lattice only positions returned")

    return name, length, dfpos


def parse_from_madx_sequence_file(filename: str) -> Tuple[str, float, pd.DataFrame]:
    """Method to parse madx seq from file to table format.

    :param str filename: filename of the file containing the sequence.
    """
    with open(filename, "r") as f:
        string = f.read()

    return parse_from_madx_sequence_string(string)


# ==============================================================================
# Table parsers
# ==============================================================================
def _parse_table_to_madx_definitions(df: pd.DataFrame) -> str:
    """
    Method to parse table to MADX sequence file definitions.

            :param pd.DataFrame df: table to parse
            :returns: Madx string for the element defintions.
    """
    # init output
    text = """"""

    df = df.drop(columns=["pos", "at"], errors="ignore")
    df = df.drop_duplicates()
    # loop over the rows of the frame
    for _, row in df.iterrows():
        # get the element family to check against allowed attrs
        keyword = row["family"]

        # get allowed attrs - to distinguish madx from elegant columns
        allowed_attrs = MADX_ATTRIBUTES[keyword].keys()

        # init line
        line = ""

        # name and element type
        line += "{:16}: {:12}, ".format(row["name"], keyword)

        # remove non attrs from columns
        row = row.drop(
            ["name", "at", "family", "end_pos", "sector", "pos"], errors="ignore"
        ).dropna()

        # add allowed madx attributes
        if len(allowed_attrs) > 0:
            attr_line = (
                ", ".join(
                    [
                        "{}:={}".format(c, row[c])
                        if c in allowed_attrs and c != "NO_CAVITY_TOTALPATH"
                        else "{}={}".format(c, str(row[c]).lower())
                        if c == "NO_CAVITY_TOTALPATH"
                        else ""
                        for c in row.index
                    ]
                )
                + ";\n"
            )
        else:
            attr_line = ";\n"
            line = line[:-2]

        line += attr_line

        # add line to text
        text += line

    return text


def _parse_table_to_madx_sequence_part(name: str, length: float, df: pd.DataFrame) -> str:
    """
    Method to parse a table to the MADX sequence part.

            :param str name: lattice name
            :param float length: lattice length
            :param pd.DataFrame df: table to parse
            :returns: Madx sequence string with positions of the elements.
    """
    # start the sequence definition
    text = "{}: SEQUENCE, L={};\n".format(name, length)

    # loop over the table rows
    for _, row in df.iterrows():
        line = "{:11}, at = {:12.6f};\n".format(row["name"], row["at"])
        text += line

    # close the sequence definition
    text += "ENDSEQUENCE;"

    return text


def parse_table_to_madx_sequence_string(name: str, length: float, df: pd.DataFrame) -> str:
    """
    Method to parse table to MADX sequence.


            :param str name: lattice name
            :param float length: lattice length
            :param pd.DataFrame df: table to parse
            :returns: Madx sequence string

    """
    # parse the element definitions
    text = _parse_table_to_madx_definitions(df)

    # parse the element positions
    text += _parse_table_to_madx_sequence_part(name, length, df)

    return text


def parse_table_to_madx_sequence_file(
    name: str, length: float, df: pd.DataFrame, filename: str
) -> None:
    """Method to parse table to madx sequence and save in file.

    :param str name: lattice name
    :param float length: lattice length
    :param pd.DataFrame df: table to parse
            :param str filename: filename of the file
    """
    save_string(parse_table_to_madx_sequence_string(name, length, df), filename)


def parse_table_to_madx_install_str(name: str, df: pd.DataFrame) -> str:
    """
    Method to parse table to MADX SEQEDIT INSTALL string.
    This can be saved to file to load via CALL or used
    directly as MADX input string using MADX().input(str) from
    cpymad package.

            :param str name: lattice name
            :param pd.DataFrame df: table with elements to install
            :returns: Madx install str
    """

    # start sequence edit
    text = "USE, SEQUENCE={};\n".format(name)
    text += "SEQEDIT, SEQUENCE = {};  \nFLATTEN;\n".format(name)
    for _, row in df.iterrows():
        line = "INSTALL, ELEMENT = {:16}, AT = {:12.6f};\n".format(row["name"], row["at"])
        text += line

    # end sequence edit
    text += "FLATTEN;\nENDEDIT;"

    return text


def parse_table_to_madx_remove_str(name: str, df: pd.DataFrame) -> str:
    """
    Method to parse a seq table to MADX SEQEDIT REMOVE string.
    This can be saved to file to load via CALL or used directly
    as MADX input string using MADX().input(str) from the cpymad
    package.

            :param str name: lattice name
            :param pd.DataFrame df: table with elements to remove
            :returns: Madx remove string
    """
    # start sequence edit
    text = "USE, SEQUENCE={};\n".format(name)
    text += "SEQEDIT, SEQUENCE = {};  \nFLATTEN;\n".format(name)
    for _, row in df.iterrows():
        line = "REMOVE, ELEMENT = {:16};\n".format(row["name"])
        text += line

    # end sequence edit
    text += "FLATTEN;\nENDEDIT;"

    return text


def parse_table_to_elegant_string(name: str, df: pd.DataFrame) -> str:
    """
    Method to transform the MADX seq table to an Elegant lte file.add()

            :param str name: lattice name
            :param pd.DataFrame df: table
            :returns: elegant lattice string
    """
    # init output
    text = """"""
    lattice_template = "{}: LINE=({})".format
    # element_template = "{}: {}, {}".format

    df = df.drop(columns=["pos", "at"], errors="ignore")
    lattice_elements = ", ".join(list(df["name"].values))
    lattice = lattice_template(name, lattice_elements)

    df = df.drop_duplicates()
    # loop over the rows of the frame
    for _, row in df.iterrows():
        # get the element family to check against allowed attrs
        keyword = TO_ELEGANT_ELEMENTS[row["family"]]

        # get allowed attrs - to distinguish madx from elegant columns
        # print(keyword)
        # print(TO_ELEGANT_ATTR)
        allowed_attrs = ELEGANT_ATTRIBUTES[keyword]

        line = ""

        # name and element type
        line += "{:16}: {:12}, ".format(row["name"], keyword)

        # remove non attrs from columns
        row = row.drop(["name", "at", "family", "end_pos", "sector"], errors="ignore").dropna()
        # nrow = [TO_ELEGANT_ATTR[c] for c in row.index if (TO_ELEGANT_ATTR[c] != "")]
        nrow = [TO_ELEGANT_ATTR[c] for c in row.index if c in allowed_attrs]
        # print(row.index)

        # add allowed madx attributes
        if len(allowed_attrs) > 0 and len(nrow) > 0:
            attr_line = (
                ", ".join(
                    [
                        "{}={:16.12f}".format(c, row[c])
                        if TO_ELEGANT_ATTR[c] in allowed_attrs and not isinstance(row[c], str)
                        else "{}={:16}".format(c, row[c])
                        if TO_ELEGANT_ATTR[c] in allowed_attrs
                        else ""
                        for c in nrow
                        # if TO_ELEGANT_ATTR[c] in allowed_attrs
                    ]
                )
                + "\n"
            )
        else:
            attr_line = "\n"
            line = line[:-2]

        line += attr_line

        # add line to text
        text += line
        # print(text)

    text += "\n\n"
    text += lattice

    return text


def parse_table_to_elegant_file(name: str, df: pd.DataFrame, filename: str) -> None:
    """Method to parse table to elegant lattice and save to file.

    :param str name: lattice name
    :param pd.DataFrame df: table
    """
    save_string(parse_table_to_elegant_string(name, df), filename)


def parse_table_to_tracy_string(name: str, df: pd.DataFrame) -> str:
    """
    Method to transform the MADX seq table to tracy lattice string.

            :param str name: lattice name
            :param pd.DataFrame df: table
            :returns: tracy lattice string
    """

    # init output
    text = """"""
    template_marker = "{}: Marker;".format
    template_bpm = "{}: Beam Position Monitor;".format
    template_drift = "{}: Drift, {};".format
    template_bend = "{}: Bending, {}, N = Nbend, Method = 4;".format
    template_quad = "{}: Quadrupole, {}, N = Nquad, Method = 4;".format
    template_sext = "{}: Sextupole, {}, N = Nsext, Method = 4;".format
    template_oct = "{}: Multipole, L = {}, HOM = (4,{}/6.0,0.0), N = Nsext, Method = 4;".format
    template_cav = "{}: Cavity, {};".format

    latname = name
    lattice_template = "{}: "
    n_elem = 10
    lattice_elements = list(df["name"].values)
    n = len(lattice_elements)
    if n >= n_elem:
        lattice_template += "\n "
    for k in range(2, n + 2):
        if (k - 1) % (n_elem + 1) == 0:
            lattice_template += "\n "
        lattice_template += lattice_elements[k - 2]
        if k < n + 1:
            lattice_template += ", "
        else:
            lattice_template += ";"
    # element_template = "{}: {}, {}".format

    df = df.drop(columns=["pos", "at"], errors="ignore")
    lattice = lattice_template.format(latname)

    df = df.drop_duplicates()
    # loop over the rows of the frame
    for _, row in df.iterrows():
        # get the element family to check against allowed attrs
        keyword = TO_TRACY_ELEMENTS[row["family"]]
        name = row["name"]

        # get allowed attrs - to distinguish madx from elegant columns
        allowed_attrs = TRACY_ATTRIBUTES[keyword]
        # print(allowed_attrs)

        # line = ""
        # name and element type
        # line += "{:16}: {:12}, ".format(row["name"], keyword)

        # remove non attrs from columns
        row = row.drop(["name", "at", "family", "end_pos", "sector"], errors="ignore").dropna()
        # print(row.index)
        # update the indices

        try:
            row.index = [TO_TRACY_ATTR[c] for c in row.index if TO_TRACY_ATTR[c] != ""]
            # row.index = [TO_TRACY_ATTR[c] for c in row.index if c in allowed_attrs]
            nrow = row.index
        except:
            nrow = [TO_TRACY_ATTR[c] for c in row.index if TO_TRACY_ATTR[c] != ""]
            # nrow = [TO_TRACY_ATTR[c] for c in row.index if c in allowed_attrs]

        if keyword == "bpm":
            line = template_bpm(name)
        elif keyword == "marker":
            line = template_marker(name)
        elif keyword == "drift":
            line = template_drift(name, "L = {}".format(row["L"]))
        elif keyword == "bend":
            new_row = {"L": row["L"]}
            new_row["T"] = np.degrees(row["T"])

            if "Roll" in nrow:
                new_row["Roll"] = np.degrees(row.get("Roll", None))

            if "Gap" in nrow:
                new_row["Gap"] = 4.0 * row.Gap * row.loc_fint

            if "T1" in nrow:
                new_row["T1"] = np.degrees(row.T1)
            if "T2" in nrow:
                new_row["T2"] = np.degrees(row.T2)
            if "K" in nrow:
                new_row["K"] = row.K

            line = template_bend(
                name,
                ", ".join(
                    [
                        "{} = {:17.15f}".format(k, v)
                        if k not in ["L", "K"]
                        else "{} = {:8.6f}".format(k, v)
                        for k, v in new_row.items()
                    ]
                ),
            )
            # line += ", N = Nbend, Method = 4;"

        elif keyword == "quad":
            line = template_quad(
                name,
                ", ".join(
                    ["{} = {:8.6f}".format(k, v) for k, v in row.items() if k in allowed_attrs]
                ),
            )
            # line += ", N = Nquad, Method = 4;"

        elif keyword == "sext":
            line = template_sext(
                name,
                ", ".join(
                    [
                        "{} = {:8.6f}".format(k, v) if k != "K" else "{} = {}/2.0".format(k, v)
                        for k, v in row.items()
                        if k in allowed_attrs
                    ]
                ),
            )
            # line += ", N = Nsext, Method = 4;"
        elif keyword == " oct1":
            line = template_oct(name, row["L"], row["K"])

        elif keyword == "cavity":
            new_row = {"L": row["L"]}
            new_row["Frequency"] = row.get("Frequency", 0.0)
            new_row["Voltage"] = row.get("Voltage", 0.0)
            new_row["phi"] = row.get("phi", 0)

            line = template_cav(
                name,
                ", ".join(
                    [
                        "{} = {:17.15f}".format(k, float(v))
                        if k not in ["L"]
                        else "{} = {:8.6f}".format(k, float(v))
                        for k, v in new_row.items()
                    ]
                ),
            )

        line += "\n"

        # add line to text
        text += line
        # print(text)
    text += "\n\n"
    text += lattice

    text += "\n\n"
    text += "ring: {};\n\n".format(latname)
    text += "cell: ring, symmetry = 1;"
    text += "\n\nend;"

    return text


def parse_table_to_tracy_file(name: str, df: pd.DataFrame, filename: str) -> None:
    """Method to transform the MADX seq table to tracy lattice and write to file.

    :param str name: lattice name
    :param pd.DataFrame df: table
    :param str filename: file where to save to
    """
    save_string(parse_table_to_tracy_string(name, df), filename)
