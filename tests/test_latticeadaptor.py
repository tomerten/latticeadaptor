# -*- coding: utf-8 -*-

"""Tests for latticeadaptor package."""

import latticeadaptor
import pandas as pd
import pytest
from latticeadaptor.parsers import (
    parse_from_madx_sequence_string,
    parse_table_to_madx_sequence_string,
)
from pandas.testing import assert_frame_equal

# test inputs and outputs
seq_str_test = [
    (
        """
    testmarker : MARKER;
    """,
        None,
        0.0,
        pd.DataFrame([{"family": "MARKER", "name": "TESTMARKER"}]),
    ),
    (
        """
    FODO: SEQUENCE, L=8;
    QF  , at = 2.000;
    ENDSEQUENCE;
    """,
        "FODO",
        8.0,
        pd.DataFrame([{"name": "QF", "pos": 2.0}]),
    ),
]

table_to_seq = [
    (
        "FODO",
        pd.DataFrame([{"family": "MARKER", "name": "TESTMARKER", "at": 2.0}]),
        8.0,
        "TESTMARKER      : MARKER      ;\nFODO: SEQUENCE, L=8.0;\nTESTMARKER , at =     2.000000;\nENDSEQUENCE;",
    ),
]


def test_version():
    assert latticeadaptor.__version__ == "0.3.1"


@pytest.mark.parametrize("string, expected_name, expected_len, expected_df", seq_str_test)
def test_parse_from_madx_seqeunce_string(string, expected_name, expected_len, expected_df):
    name, length, df = parse_from_madx_sequence_string(string)

    print(df)
    assert name == expected_name
    assert length == expected_len
    assert_frame_equal(df, expected_df, check_like=True)


@pytest.mark.parametrize("name, df, length, string", table_to_seq)
def test_parse_to_madx_seqeunce_string(name, df, length, string):
    newstring = parse_table_to_madx_sequence_string(name, length, df)
    assert string == newstring


# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (otherwise all tests are normally run with pytest)
# Make sure that you run this code with the project directory as CWD, and
# that the source directory is on the path
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_parse_to_madx_seqeunce_string

    print("__main__ running", the_test_you_want_to_debug)
    the_test_you_want_to_debug()
    print("-*# finished #*-")

# eof
