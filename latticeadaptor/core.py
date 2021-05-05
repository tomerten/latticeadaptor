# -*- coding: utf-8 -*-

"""
Module latticeadaptor.core 
=================================================================

A module containing the main class to operate on accelerator lattices.

"""

# your imports here ...
import queue
from copy import deepcopy


class LatticeAdaptor:
    """Class to convert lattices."""

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", None)
        self.len = kwargs.get("len", 0.0)
        self.table = kwargs.get("table", None)
        self.filename = kwargs.get("file", None)
        self.inputstr = kwargs.get("string", None)

        # roll back
        self.history = queue.LifoQueue()

def greet(to=''):
	"""Say "Hello <to>!".
	
	:param str to: whom you want to say hello to.
	"""
	f"Hello {to}!"
