# -*- coding: utf-8 -*-

"""
Module latticeadaptor.core 
=================================================================

A module containing the main class to operate on accelerator lattices.

"""

import os

# your imports here ...
import queue
import re
from copy import deepcopy
from typing import Tuple

import numpy as np
import pandas as pd
from latticeconstructor.core import LatticeBuilderLine
from latticeconstructor.parse import parse_from_string

from .parsers import (
    _parse_table_to_madx_definitions,
    parse_from_madx_sequence_file,
    parse_from_madx_sequence_string,
    parse_table_to_elegant_file,
    parse_table_to_elegant_string,
    parse_table_to_madx_install_str,
    parse_table_to_madx_remove_str,
    parse_table_to_madx_sequence_file,
    parse_table_to_madx_sequence_string,
    parse_table_to_tracy_file,
    parse_table_to_tracy_string,
    save_string,
)
from .utils import install_start_end_marker


class LatticeAdaptor:
    """Class to convert lattices."""

    def __init__(self, **kwargs):
        # roll back
        self.history = queue.LifoQueue()
        self.name = kwargs.get("name", None)
        self.len = kwargs.get("len", 0.0)
        self._table = None
        self.table = kwargs.get("table", None)
        self.filename = kwargs.get("file", None)
        self.inputstr = kwargs.get("string", None)
        self.builder = LatticeBuilderLine()

    @property
    def table(self):
        return self._table

    @table.setter
    def table(self, value):
        # roll back
        self.history.put((deepcopy(self.name), deepcopy(self.len), deepcopy(self.table)))

        self._table = value

    def load_from_file(self, filename: str, ftype: str = "lte") -> None:
        """Load data from file.

        Parameters
        ----------
        filename : str
            filename of the file to load lattice from
        ftype : str, optional
            lattice format to load from, currenlty lte and madx allowed, by default "lte"
        """

        # use the latticebuilder to build the table
        self.builder.load_from_file(filename, ftype)
        self.builder.build_table()

        # extract the relevant info
        # Better way?
        try:
            self.name = self.builder.name
        except Exception:
            self.name = "ring"
            print("Lattice name has been autoset - check if value is ok.")
        self.table = self.builder.table

        # length is last element center pos + half the  length
        with open(filename, "r") as f:
            txt = f.read()

        length = re.findall("sequence\s*,\s*\s*[lL]\s*[:]?=\s*(\d+[.]?\d+)", txt)

        if (ftype == "madx") and (len(length) != 0):
            self.len = length[0]
        if not self.len:
            self.len = (
                self.table.tail(1)["at"].values[-1] + self.table.tail(1)["L"].values[-1] / 2.0
            )
            print(
                "Length has been autoset to {} - check if value is ok - otherwise update it.".format(
                    self.len
                )
            )

    def load_from_string(self, string: str, ftype: str = "lte") -> None:
        """Load data from string.

        Parameters
        ----------
        string : str
            string input
        ftype : str, optional
            format of string, currenlty supported are lte and madx, by default "lte"
        """
        # roll back
        self.history.put((deepcopy(self.name), deepcopy(self.len), deepcopy(self.table)))

        # write to temp file
        with open("templatticestringfile.tmp", "w") as f:
            f.write(string)

        # use load from file method
        self.load_from_file("templatticestringfile.tmp", ftype)

    def parse_table_to_madx_sequence_string(self) -> str:
        """Parse table to MADX sequence and return it as a string.

        Returns
        -------
        str
            MADX sequence
        """
        return parse_table_to_madx_sequence_string(self.name, self.len, self.table)

    def parse_table_to_madx_sequence_file(self, filename: str) -> None:
        """Parse table to MADX sequence and write it to file.

        Parameters
        ----------
        filename : str
            filename where the seq will be written to.
        """
        parse_table_to_madx_sequence_file(self.name, self.len, self.table, filename)

    def parse_table_to_elegant_string(self) -> str:
        """Parse table to Elegant lattice and return it as a string.

        Returns
        -------
        str
            Elegant Lattice
        """
        self.add_drifts()
        return parse_table_to_elegant_string(self.name, self.table)

    def parse_table_to_elegant_file(self, filename: str) -> None:
        """Parse table to Elegant lattice and write it to file.

        Parameters
        ----------
        filename : str
            filename where the seq will be written to.
        """
        self.add_drifts()

        parse_table_to_elegant_file(self.name, self.table, filename)

    def parse_table_to_tracy_string(self) -> str:
        """Parse table to Tracy lattice and return it as a string.

        Returns
        -------
        str
            Tracy Lattice
        """
        return parse_table_to_tracy_string(self.name, self.table)

    def parse_table_to_tracy_file(self, filename: str) -> None:
        """Parse table to Tracy lattice and write it to file.

        Parameters
        ----------
        filename : str
            filename where the seq will be written to.
        """
        parse_table_to_tracy_file(self.name, self.table, filename)

    def madx_sequence_add_start_end_marker_string(self) -> str:
        """Return MADX string to install marker at start and end of the lattice.

        Returns
        -------
        str
            MADX install string that can be run with cpymad.
        """
        return install_start_end_marker(self.name, self.len)

    def parse_table_to_madx_install_str(self) -> str:
        """Method to generate a MADX install element string based on the table.
        This string can be used by cpymad to install new elements.

        Returns
        -------
        str
            Install element string input for MADX.
        """
        return parse_table_to_madx_install_str(self.name, self.table)

    def parse_table_to_madx_remove_str(self) -> str:
        """Method to generate a MADX remove element string based on the table.
        This string can be used by cpymad to remove elements.add()

        Returns
        -------
        str
            Remove element string input for MADX.
        """
        return parse_table_to_madx_remove_str(self.name, self.table)

    def madx_sequence_save_string(self, filename: str) -> str:
        """Method to generate string input for MADX to save the lattice
        to sequence.add()

        Parameters
        ----------
        filename : str
            filename of where to write the sequence to.

        Returns
        -------
        str
            save sequence string.
        """
        return "SAVE, SEQUENCE={}, file='{}';".format(self.name, filename)

    def add_drifts(self):
        """Method to add back drifts to sequence."""
        self.history.put((deepcopy(self.name), deepcopy(self.len), deepcopy(self.table)))

        df = self.table.copy()
        df.reset_index(inplace=True, drop=True)
        name = "D"
        family = "DRIFT"

        df.loc[df.L.isna(), "L"] = 0
        if "pos" not in df.columns:
            df["pos"] = df["at"]
        newrows = []
        ndrift = 0
        for i, row in df.iterrows():
            # add the row
            newrows.append(pd.DataFrame(row).T)

            # check if next row
            if i < len(df) - 1:
                # check if next row pos is not equal to the current
                nextrow = df.loc[i + 1]
                # print(
                #    row["pos"],
                #    nextrow["pos"],
                #    nextrow["pos"] > row.pos,
                #    nextrow["pos"] - (nextrow["L"] / 2.0) > row.pos + row.L / 2.0,
                # )
                if nextrow["pos"] - (nextrow["L"] / 2.0) > row.pos + row.L / 2.0:
                    ndrift += 1
                    newrow = {}
                    newrow["name"] = name + str(ndrift)
                    newrow["family"] = family
                    newrow["L"] = np.round(
                        (nextrow["pos"] - nextrow["L"] / 2.0) - (row["pos"] + row["L"] / 2.0), 6
                    )
                    newrow["pos"] = (row["pos"] + row["L"] / 2.0) + (newrow["L"] / 2.0)
                    newrows.append(pd.Series(newrow).to_frame().T)

        # if lattice length is longer than end of last element there is still a drift
        if nextrow["pos"] + nextrow["L"] / 2.0 < self.len:
            newrow = {}
            newrow["name"] = name + str(ndrift)
            newrow["family"] = family
            newrow["L"] = np.round(self.len - nextrow["pos"], 6)
            newrow["pos"] = (row["pos"] + row["L"] / 2.0) + (newrow["L"] / 2.0)
            newrows.append(pd.Series(newrow).to_frame().T)

        self.table = (pd.concat(newrows)).reset_index(drop=True)

    def parse_table_to_madx_line_string(self) -> str:
        """Method to convert the table to a MADX line definition lattice.

        Returns
        -------
        str
            MADX lattice definition string.
        """
        self.add_drifts()
        defstr = _parse_table_to_madx_definitions(self.table)
        linestr = "{}: LINE=({});".format(
            self.name,
            ",\n\t\t".join(
                [",".join(c) for c in list(self.chunks(self.table.name.to_list(), 20))]
            ),
        )
        return defstr + "\n\n" + linestr

    @staticmethod
    def chunks(lst: list, n: int):
        """Yields successive n-sized chunks from a list.

        Parameters
        ----------
        lst : list
            list
        n : int
            chunck size

        Yields
        -------
        list
            chunk
        """
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def parse_table_to_madx_line_file(self, filename: str):
        """Method to convert table to madx line def lattice file string and write to file.

        Parameters
        ----------
        filename : str
            file to write to
        """
        save_string(self.parse_table_to_madx_line_string(), filename)

    def get_quad_strengths(self) -> dict:
        """Method to return quadrupole strengths as a dict.

        Returns:
        --------
        dict
            dictionary with quadrupole strengths
        """
        return (
            self.table.loc[self.table.family == "QUADRUPOLE", ["name", "K1"]]
            .set_index("name", drop=True)
            .to_dict()["K1"]
        )

    def get_sext_strengths(self) -> dict:
        """Method to return sextupole strengths as a dict

        Returns:
        --------
        dict
            dictionary with sextupole strengths
        """
        if "SEXTUPOLE" in self.table.family.values:
            return (
                self.table.loc[self.table.family == "SEXTUPOLE", ["name", "K2"]]
                .set_index("name", drop=True)
                .to_dict()["K2"]
            )
        else:
            return {}

    def load_strengths_to_table(self, strdc: dict, col: str) -> None:
        """Method to load a dictionary with strength settings to the table.
        The col attribute is where the strengths will be loaded to.

        Parameters
        ----------
        strdc : dict
            strength settings dict
        col : str
            column where to write the settings
        """
        self.history.put((deepcopy(self.name), deepcopy(self.len), deepcopy(self.table)))

        for k, v in strdc.items():
            self.table.loc[self.table["name"] == k, col] = v

    def compare_seq_center_positions(self, seqfile2: str) -> Tuple[pd.DataFrame]:
        """Compares the center positions of elements of the lattice in the table
        with another lattice table.

        Parameters
        ----------
        seqfile2 : str
            MADX sequence file of the second lattice.add()

        Returns
        -------
        Tuple[pd.DataFrame]
            returns two dataframes, the first is with equal positions, the second with the different positions
        """
        # assert os.path.isfile(seqfile1)
        assert os.path.isfile(seqfile2)

        # name1, len1, df1 = parse_from_madx_sequence_file(seqfile1)
        name2, len2, df2 = parse_from_madx_sequence_file(seqfile2)

        table1 = self.table[["name", "pos"]]
        table2 = df2[["name", "pos"]]

        eq = pd.merge(table1, table2, on=["pos"], how="inner")
        diff = table1[~table1["pos"].isin(table2["pos"])]

        return eq, diff

    def update_table(self):
        """Reload the table from the builder table."""
        # updated history
        self.history.put((deepcopy(self.name), deepcopy(self.len), deepcopy(self.table)))

        self.builder.positions = None
        self.builder.build_table()

        # extract the relavant info
        self.name = self.builder.name
        self.table = self.builder.table

        # length is last element center pos + half the  length
        print("Length has been autoset - check if value is ok - otherwise update it.")
        self.len = (
            self.builder.table.tail(1)["at"].values[-1]
            + self.builder.table.tail(1)["L"].values[-1] / 2.0
        )

    def undo(self):
        """Undo previous change."""
        if not self.history.empty():
            old = self.history.get()
            self.name, self.length, self.table = old
        else:
            print("No previous states available")
