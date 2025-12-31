.. BOAT documentation master file, created by
   sphinx-quickstart on Tue Dec 31 15:44:50 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BOAT-Torch Documentation
===================================

**BOAT** is a task-agnostic, gradient-based **Bi-Level Optimization (BLO)** Python library that focuses on abstracting the key BLO process into modular, flexible components. It enables researchers and developers to tackle learning tasks with hierarchical nested nature by providing customizable and diverse operator decomposition, encapsulation, and combination. BOAT supports specialized optimization strategies, including second-order or first-order, nested or non-nested, and with or without theoretical guarantees, catering to various levels of complexity.

.. image:: _static/flow.gif
   :alt: BOAT Framework
   :width: 1000px
   :align: center

In this section, we explain the core components of BOAT, how to install the PyTorch version, and how to use it for your optimization tasks. The main contents are organized as follows.

.. toctree::
   :maxdepth: 2
   :caption: Installation Guide:

   description.md
   install_guide.md
   boat_torch.rst

Running Example
----------------------------

The running examples of data hypercleaning, meta learning and l2 regularization are organized as follows.

.. toctree::
   :maxdepth: 2
   :caption: Example:

   data_hyper_cleaning_example.md
   l2_regularization_example.md
   meta_learning_example.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`