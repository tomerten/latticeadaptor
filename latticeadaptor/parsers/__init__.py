# -*- coding: utf-8 -*-

"""
Module latticeadaptor.parsers 
=================================================================

A module

"""

# your imports here ...
from abc import ABC
from pathlib import Path

import pandas as pd
from lark import Lark, Transformer, v_args
from lark.exceptions import LarkError

BASE_DIR = Path(__file__).resolve().parent
with (BASE_DIR / "../lark/Madx.lark").open() as file:
    MADX_PARSER = Lark(file, parser="lalr", maybe_placeholders=True)
    file.seek(0)


@v_args(inline=True)
class AbstractSequenceFileTransformer(ABC, Transformer):
    def transform(self, tree):
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


def parse_from_madx_sequence_string(string: str) -> (str, float, pd.DataFrame):
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


def parse_from_madx_sequence_file(filename: str) -> (str, float, pd.DataFrame):
    """Method to parse madx seq from file to table format.

    :param str filename: filename of the file containing the sequence.
    """
    with open(filename, "r") as f:
        string = f.read()

    return parse_from_madx_sequence_string(string)
