*********
Tutorials
*********

1. Loading Lattices
===================
Loading a lattice from a Madx sequence file.

.. code-block:: python

    from latticeadaptor.core import LatticeAdaptor
    la = LatticeAdaptor()
    lat = "lattice.seq"
    la.load_from_madx_sequence_file(lat)


Loading a lattice from a Madx sequence string.

.. code-block:: python

    from latticeadaptor.core import LatticeAdaptor
    latticestring = """
    QF : QUADRUPOLE, L := 0.50 , K1 :=  1.00;
    QD : QUADRUPOLE, L := 1.00 , K1 := -1.00;
    D1 : DRIFT, L := 1.00;
    D2 : DRIFT, L := 1.00;

    FODO: SEQUENCE, L=4.00;
    QF, at = 0.25;
    D1, at = 1.00;
    QD, at = 2.00;
    D2, at = 3.00;
    QF, at = 3.75;
    ENDSEQUENCE;
    """
    la = LatticeAdaptor()
    la.load_from_madx_sequence_string(latticestring)


Loading manually a lattice name, lattice length and lattice table.

.. code-block:: python

    from latticeadaptor.core import LatticeAdaptor
    import pandas as pd
    latticename = 'FODO'
    latticelen  = 4.00
    latticetable = pd.DataFrame([
        {'name': 'QF','pos': 0.25,'family': 'QUADRUPOLE','L': 0.5,'K1': 1.0,'ANGLE': 0.0,'at': 0.25},
        {'name': 'D1','pos': 1.0,'family': 'DRIFT','L': 1.0,'at': 1.0},
        {'name': 'QD','pos': 2.0,'family': 'QUADRUPOLE','L': 1.0,'K1': -1.0,'at': 2.0},
        {'name': 'D2','pos': 3.0,'family': 'DRIFT','L': 1.0,'at': 3.0},
        {'name': 'QF','pos': 3.75,'family': 'QUADRUPOLE','L': 0.5,'K1': 1.0,'ANGLE': 0.0,'at': 3.75}
        ])
    la = LatticeAdaptor()
    la.name  = latticename
    la.len   = latticelen
    la.table = latticetable

2. Saving lattices - converting
===============================

3.1 Madx
--------

Returning the current table as a Madx sequence string.

.. code-block:: python

    la.parse_table_to_madx_sequence_string()

Saving this sequence directly to file.

.. code-block:: python

    la.parse_table_to_madx_sequence_file("lattice.seq")

Generate Madx input string to save a lattice to a sequence file.

.. code-block:: python

    la.madx_sequence_save_string()

.. note::

    The package can also save a table to a Madx lattice using the `line` command. The 
    method that does this automatically adds drifts to the table and iteratively names them.

.. code-block:: python

    la.parse_table_to_madx_line_string()
    la.parse_table_madx_line_file('linelattice.madx')



3.2 Elegant
-----------

Returning the current table as a Elegant lattice string.

.. code-block:: python

    la.parse_table_to_elegant_string()

Saving this sequence directly to file.

.. code-block:: python

    la.parse_table_to_elegant_file("lattice.lte")

3.3 Tracy
---------

Returning the current table as a Tracy lattice string.

.. code-block:: python

    la.parse_table_to_tracy_string()

Saving this sequence directly to file.

.. code-block:: python

    la.parse_table_to_tracy_file("lattice.lat")

4. Basic lattice operations
===========================

4.1 Markers
-----------
Generate Madx input string to add lattice start and end marker.

.. code-block:: python

    la.madx_sequence_add_start_end_marker_string()

4.2 Drifts
----------
Add drifts back to the sequence.

.. code-block:: python

    la.add_drifts()

4.3 Settings
------------

Get QUADRUPOLE and SEXTUPOLE settings.

.. code-block:: python

    la.get_quad_strengths()
    la.get_sext_strengths()

Load a dictionary with settings to the table. Next to the actual settings dictionary (name: set_value)
one also needs to provide the argument or column name that is being set. For example, as shown below, 
for a quadrupole one can set the `K1` column.

.. code-block:: python

    settings_dict = {'Q1' : 1.523}
    la.load_strengths_to_table(settings_dict, 'K1'}


5. Comparing lattices
=====================

When working with accelerator lattices in various formats it is often difficult to keep
track of if all elements are still where they need to be after some editing. The package 
provides a method to check which elements center positions are the same and which are different.

.. code-block:: python

    la.compare_seq_center_positions('lattice2.seq')

Another common thing to do is to compare element settings for different lattices. Once the
settings are extraced (for example by the ``get_quad_strengths`` or by using pandas DataFrame
filtering and extraction on the lattice table) one can compare them with:

.. code-block:: python

    quad_set1 = la1.get_quad_strengths()
    quad_set2 = la2.get_quad_strengths()
    compare_seq_center_positions(quad_set1, quad_set2, threshold=1)


.. note::

    The threshold value is used to highlight the differences. When the 
    difference is larger than the threshold the entries will be highlighted
    in red. Equal values are in green and non-equal values but with differences
    below the threshold will be in orange.


The ``Beamlinegraph_compare_from_seq_files`` allows for a graphical check of the 
alignment of the lattice elements.

.. plot::

    from latticeadaptor.utils import Beamlinegraph_compare_from_seq_files
    Beamlinegraph_compare_from_seq_files('fodob.seq','fodo.seq')



6. Element Plotting Example
===========================

.. note::

    The relative size of the elements in the plots below is a representation of
    their relative strength settings. 

6.1 FODO
--------

.. plot::
    
    from latticeadaptor.core import LatticeAdaptor
    from latticeadaptor.utils import Beamlinegraph_from_seq_file
    madxseqsymm = """
    QF : QUADRUPOLE, L := 0.50 , K1 :=  1.00;
    QD : QUADRUPOLE, L := 1.00 , K1 := -1.00;
    D1 : DRIFT, L := 1.00;
    D2 : DRIFT, L := 1.00;

    FODO: SEQUENCE, L=4.00;
    QF, at = 0.25;
    D1, at = 1.00;
    QD, at = 2.00;
    D2, at = 3.00;
    QF, at = 3.75;
    ENDSEQUENCE;
    """
    la = LatticeAdaptor()
    la.load_from_madx_sequence_string(madxseqsymm)
    la.parse_table_to_madx_sequence_file('fodo.seq')
    Beamlinegraph_from_seq_file('fodo.seq')


6.2. FODO WITH BEND
-------------------

.. plot::

    from latticeadaptor.core import LatticeAdaptor
    from latticeadaptor.utils import Beamlinegraph_from_seq_file
    madxseqsymm = """
    QF: QUADRUPOLE, L=0.5,K1=0.2; 
    QD: QUADRUPOLE, L=1.0,K1=-0.2; 
    B: SBEND, L=1.0, ANGLE=15.0; 
    FODO: SEQUENCE, L=12.0;
    QF, at = 0.25;
    B,  at = 3.00;
    QD, at = 6.00;
    B,  at = 9.00;
    QF, at = 11.75;
    ENDSEQUENCE;
    """
    la = LatticeAdaptor()
    la.load_from_madx_sequence_string(madxseqsymm)
    la.parse_table_to_madx_sequence_file('fodob.seq')
    Beamlinegraph_from_seq_file('fodob.seq')

6.3 Twiss plot
--------------

The implemented twiss plot method currently only works with a twiss object produced 
by running the Twiss command using `cpymad <https://github.com/hibtc/cpymad>`_.

.. plot:: 

    from latticeadaptor.utils import twissplot
    from cpymad.madx import Madx
    madx = Madx(stdout=False)
    madx.command.beam(particle='electron',energy=1.7)
    madx.call(file='fodo.seq')
    madx.use(sequence='FODO')
    twiss = madx.twiss()
    twissplot(twiss)

7. More advanced editing
========================

At light sources one often needs to split the dipoles to insert markers for the 
beam ports in order to extract the exact lattice functions at these locations. The package
therefor provides a method to generate dipole splits.

.. code-block:: python

    from latticeadaptor.utils import dipole_split_angles_to_dict
    dipolename = 'Q1M1'
    dipolelen  = 6.00
    dipoleanglerad = 0.098 
    anglelistdeg = [1.4,2.5,6.4]
    split_dict = dipole_split_angles_to_dict(dipolename,dipolelen,dipoleanglerad,anglelistdeg,verbose=True)


This dictionary can now be used to update the table.

.. code-block:: python

    from latticeadaptor.utils import split_dipoles
    la.table = split_dipoles(la.table, split_dict ,dipoleanglerad/2)


8. Undo
=======

Sometimes we make mistakes, do not worry there is an ``undo`` method.

.. code-block:: python

    la.add_drifts()
    #ohh no no - not what I wanted
    la.undo()