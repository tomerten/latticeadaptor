#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `latticeadaptor` package."""

import latticeadaptor.utils
import pytest


def test_list_rotate():
    expected = [3, 1, 2]
    actual = latticeadaptor.utils.rotate([1, 2, 3], 1)
    assert actual == expected


# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_list_rotate

    the_test_you_want_to_debug()
    print("-*# finished #*-")
# ==============================================================================
