
# Introduction
[![PyPI version](https://badge.fury.io/py/boat-jit.svg)](https://badge.fury.io/py/boat-jit)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/callous-youth/BOAT/workflow.yml)
[![codecov](https://codecov.io/github/callous-youth/BOAT/graph/badge.svg?token=0MKAOQ9KL3)](https://codecov.io/github/callous-youth/BOAT)
[![pages-build-deployment](https://github.com/callous-youth/BOAT/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/callous-youth/BOAT/actions/workflows/pages/pages-build-deployment)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/callous-youth/BOAT)
![GitHub top language](https://img.shields.io/github/languages/top/callous-youth/BOAT)
![GitHub language count](https://img.shields.io/github/languages/count/callous-youth/BOAT)
![Python version](https://img.shields.io/badge/python-3.8%2B-blue)
![license](https://img.shields.io/badge/license-MIT-000000.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)


**BOAT** is a compositional **O**per**A**tion-level **T**oolbox for gradient-based **B**LO.
Unlike existing libraries that typically encapsulate fixed solver routines, BOAT factorizes the BLO workflow into **atomic, reusable primitives**. Through a unified constraint reconstruction perspective, it empowers researchers to **automatically compose** over **85+ solver variants** from a compact set of **17 gradient operations**.

- **[PyTorch-based](https://github.com/callous-youth/BOAT)**: An efficient and widely-used version.
- **[Jittor-based](https://github.com/callous-youth/BOAT/tree/boat_jit)**: An accelerated version for high-performance tasks with full implementation of the BLO algorithms in the PyTorch version of *BOAT*.
- **[MindSpore-based](https://github.com/callous-youth/BOAT/tree/boat_ms)**: Incorporating the latest first-order optimization strategies to support emerging application scenarios.


BOAT is designed to offer robust computational support for a broad spectrum of BLO research and applications, enabling innovation and efficiency in machine learning and computer vision.


## 🔑 Key Features

* **🧩 Compositional Operation-Level Abstraction**: Deconstructs BLO solvers into three modular stages: *Gradient Mapping (GM)*, *Numerical Approximation (NA)*, and *First-Order (FO)*.
* **⚡ Accelerated JIT Execution**: Built on Jittor, enabling meta-operator fusion and high-performance execution on NVIDIA GPUs.
* **🏭 Generative Solver Construction**: Supports dynamic serialization of operations. Users can recover classical algorithms or discover **novel hybrid solvers** simply by changing configurations.
* **🛠 Configuration-Driven**: Define complex optimization strategies via simple `JSON` configurations, decoupling algorithmic logic from model definitions.
* **✅ Comprehensive Testing**: Achieves **99% code coverage** through rigorous testing with **pytest**, ensuring software robustness.

##  🚀 **Why BOAT?**
Existing automatic differentiation (AD) tools primarily focus on specific optimization strategies, such as explicit or implicit methods, and are often targeted at meta-learning or specific application scenarios, lacking support for algorithm customization. 

In contrast, **BOAT** expands the landscape of Bi-Level Optimization (BLO) applications by supporting a broader range of problem-adaptive operations. It bridges the gap between theoretical research and practical deployment, offering unparalleled flexibility to design, customize, and accelerate BLO techniques.


## 🌍 Applications

BOAT covers a wide spectrum of BLO applications, categorized by the optimization target:

* **Data-Centric**: Data Hyper-Cleaning, Synthetic Data Reweighting, Diffusion Model Guidance.
* **Model-Centric**: Neural Architecture Search (NAS), LLM Prompt Optimization, Parameter Efficient Fine-Tuning (PEFT).
* **Strategy-Centric**: Meta-Learning, Hyperparameter Optimization (HO), Reinforcement Learning from Human Feedback (RLHF).



## 🚩 **Related Operations**

### **Gradient Mapping Operation Library (GM-OL)**
- [Towards gradient-based bilevel optimization with non-convex followers and beyond (DI)](https://proceedings.neurips.cc/paper_files/paper/2021/file/48bea99c85bcbaaba618ba10a6f69e44-Paper.pdf)
- [Averaged Method of Multipliers for Bi-Level Optimization without Lower-Level Strong Convexity(DM)](https://proceedings.mlr.press/v202/liu23y/liu23y.pdf)
- [A General Descent Aggregation Framework for Gradient-based Bi-level Optimization (GDA)](https://arxiv.org/abs/2102.07976)
- [Bilevel Programming for Hyperparameter Optimization and Meta-Learning (NGD)](http://export.arxiv.org/pdf/1806.04910)

### **Numerical Approximation Operation Library (NA-OL)**
- [Hyperparameter optimization with approximate gradient (CG)](https://arxiv.org/abs/1602.02355)
- [Optimizing millions of hyperparameters by implicit differentiation (NS)](http://proceedings.mlr.press/v108/lorraine20a/lorraine20a.pdf)
- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (IAD)](https://arxiv.org/abs/1703.03400)
- [On First-Order Meta-Learning Algorithms (FOA)](https://arxiv.org/abs/1703.03400)
- [Bilevel Programming for Hyperparameter Optimization and Meta-Learning (RAD)](http://export.arxiv.org/pdf/1806.04910)
- [Truncated Back-propagation for Bilevel Optimization (RGT)](https://arxiv.org/pdf/1810.10667.pdf)
- [DARTS: Differentiable Architecture Search (FD)](https://arxiv.org/pdf/1806.09055.pdf)
- [Towards gradient-based bilevel optimization with non-convex followers and beyond (PTT)](https://proceedings.neurips.cc/paper_files/paper/2021/file/48bea99c85bcbaaba618ba10a6f69e44-Paper.pdf)
- [Learning With Constraint Learning: New Perspective, Solution Strategy and Various Applications (IGA)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10430445)

### **First-Order Operation Library (FO-OL)**
- [BOME! Bilevel Optimization Made Easy: A Simple First-Order Approach (VFO)](https://proceedings.neurips.cc/paper_files/paper/2022/file/6dddcff5b115b40c998a08fbd1cea4d7-Paper-Conference.pdf)
- [A Value-Function-based Interior-point Method for Non-convex Bi-level Optimization (VSO)](http://proceedings.mlr.press/v139/liu21o/liu21o.pdf)
- [On Penalty-based Bilevel Gradient Descent Method (PGDO)](https://proceedings.mlr.press/v202/shen23c/shen23c.pdf)
- [Moreau Envelope for Nonconvex Bi-Level Optimization: A Single-loop and Hessian-free Solution Strategy (MESO)](https://arxiv.org/pdf/2405.09927)


## 📜 **License**

MIT License

Copyright (c) 2025 Yaohua Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



