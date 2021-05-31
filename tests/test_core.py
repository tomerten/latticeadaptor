#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `latticeadaptor` package."""

import pytest
from latticeadaptor.core import LatticeAdaptor


def test_to_elegant():
    la = LatticeAdaptor()
    la.builder.name = "ring"
    la.load_from_string(
        """
O1 : KOCT, L=0.05, K3=0.0

ring: LINE = (O1,O1)
""",
        ftype="lte",
    )
    print(la.parse_table_to_elegant_string())


# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
# ==============================================================================
if __name__ == "__main__":
    #    the_test_you_want_to_debug = test_greet

    #    the_test_you_want_to_debug()
    print("-*# finished #*-")
# ==============================================================================
