.. BOAT documentation master file, created by
   sphinx-quickstart on Tue Dec 31 15:44:50 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BOAT-MindSpore Documentation
=======================================

**BOAT** is a compositional, gradient-based **Bi-Level Optimization (BLO)** library that abstracts the core BLO pipeline into modular, flexible components. In this **MindSpore-based** implementation, BOAT is seamlessly integrated into MindSpore’s computation graph and heterogeneous hardware ecosystem (Ascend/GPU/CPU), enabling researchers and developers to build hierarchical learning tasks with customizable operator decomposition, encapsulation, and composition.

.. image:: _static/flow.gif
   :alt: BOAT Framework
   :width: 800px
   :align: center

In this section, we explain the core components of BOAT, how to install the MindSpore version of BOAT, and how to use it for your optimization tasks. The main contents are organized as follows.

.. toctree::
   :maxdepth: 2
   :caption: Installation Guide:

   description.md
   install_guide.md
   boat_ms.rst

Running Example
---------------

The running example of l2 regularization is organized as follows.

.. toctree::
   :maxdepth: 2
   :caption: Example:

   l2_regularization_example.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`