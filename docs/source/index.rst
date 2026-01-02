.. BOAT documentation master file, created by
   sphinx-quickstart on Tue Dec 31 15:44:50 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BOAT-Jittor Documentation
============================================

**BOAT** is a compositional, gradient-based **Bi-Level Optimization (BLO)** Python library that abstracts BLO into modular and flexible components, enabling efficient modeling of hierarchical and nested learning problems. It supports a wide spectrum of optimization settings, including first- and second-order methods, nested or non-nested formulations, with or without theoretical guarantees. This repository provides the **Jittor-based** implementation (boat-jit), leveraging Jittor’s JIT compilation and efficient CUDA/cuDNN backends to accelerate large-scale gradient-based BLO experiments.

.. image:: _static/flow.gif
   :alt: BOAT Framework
   :width: 800px
   :align: center

In this section, we explain the core components of BOAT, how to install the Jittor version, and how to use it for your optimization tasks. The main contents are organized as follows.

.. toctree::
   :maxdepth: 2
   :caption: Installation Guide:

   description.md
   install_guide.md
   boat_jit.rst

Running Example
----------------------------

The running example of l2 regularization is organized as follows.

.. toctree::
   :maxdepth: 2
   :caption: Example:

   l2_regularization_example.md


Indices and tables
==========================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
