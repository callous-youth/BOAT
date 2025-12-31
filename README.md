<h1 align="center">
  <img src="https://raw.githubusercontent.com/callous-youth/BOAT/refs/heads/main/_static/logo.jpg" alt="BOAT" width="50%" align="top">
</h1>
<p align="center">
  <b>A Compositional Operation Toolbox for Gradient-based Bi-Level Optimization</b><br>
  <br>
  <a href="https://boat.readthedocs.io/en/latest/index.html">Home</a> |
  <a href="https://boat.readthedocs.io/en/latest/install_guide.html#installation">Installation</a> |
  <a href="https://boat.readthedocs.io/en/latest/boat_ms.html">Docs</a> |
  <a href="https://boat.readthedocs.io/en/latest/install_guide.html#how-to-use-boat">Tutorials</a> |
  <a href="https://boat.readthedocs.io/en/latest/index.html#running-example">Examples</a>
</p>

<div align="center">

[![PyPI version](https://badge.fury.io/py/boat-ms.svg)](https://badge.fury.io/py/boat-ms)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/callous-youth/BOAT/workflow.yml)
[![codecov](https://codecov.io/github/callous-youth/BOAT/graph/badge.svg?token=0MKAOQ9KL3)](https://codecov.io/github/callous-youth/BOAT)
[![pages-build-deployment](https://github.com/callous-youth/BOAT/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/callous-youth/BOAT/actions/workflows/pages/pages-build-deployment)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/callous-youth/BOAT)
![GitHub top language](https://img.shields.io/github/languages/top/callous-youth/BOAT)
![GitHub language count](https://img.shields.io/github/languages/count/callous-youth/BOAT)
![Python version](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-000000.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

</div>

**BOAT** is a compositional **O**per**A**tion-level **T**oolbox for gradient-based **B**LO.

Unlike existing libraries that typically encapsulate fixed solver routines, BOAT factorizes the BLO workflow into **atomic, reusable primitives**. Through a unified constraint reconstruction perspective, it empowers researchers to **automatically compose** solvers via simple configuration.

This is the **MindSpore-based** version of BOAT (`boat-ms`). It is seamlessly integrated into MindSpore’s computation graph and heterogeneous hardware ecosystem (**Ascend/GPU/CPU**), incorporating the latest **First-Order (FO)** optimization strategies to support emerging large-scale application scenarios.

BOAT supports unified execution across backends via separate branches:
- **[PyTorch-based](https://github.com/callous-youth/BOAT)**: The widely-used standard version.
- **[Jittor-based](https://github.com/callous-youth/BOAT/tree/boat_jit)**: Accelerated version with meta-operators.

<p align="center">
  <a href="https://github.com/callous-youth/BOAT">
    <img src="https://github.com/callous-youth/BOAT/raw/main/_static/BOAT.png" alt="BOAT Structure" width="90%" align="top">
  </a>
</p>

## 🔑 Key Features

* **🧩 First-Order Operation Specialization**: Specifically optimized for **First-Order (FO)** BLO methods, enabling fast prototyping on Ascend hardware.
* **🚀 Heterogeneous Hardware Support**: Native compatibility with Huawei **Ascend (NPU)**, GPU, and CPU backends.
* **🏭 Generative Solver Construction**: Supports dynamic serialization of operations. Users can switch between different FO strategies (e.g., VFO, MESO) simply by changing configurations.
* **🛠 Configuration-Driven**: Define complex optimization strategies via simple `JSON` configurations (`boat_config` & `loss_config`), decoupling algorithmic logic from model definitions.
* **✅ Comprehensive Testing**: Achieves **99% code coverage** through rigorous testing, ensuring software robustness.

##  🚀 **Why BOAT?**
Existing automatic differentiation (AD) tools primarily focus on specific optimization strategies, such as explicit or implicit methods, and are often targeted at meta-learning or specific application scenarios, lacking support for algorithm customization. 

In contrast, **BOAT** expands the landscape of Bi-Level Optimization (BLO) applications by supporting a broader range of problem-adaptive operations. It bridges the gap between theoretical research and practical deployment, offering unparalleled flexibility to design, customize, and accelerate BLO techniques.

## 📚 Supported Operation Libraries

<div align="center">

The MindSpore version of BOAT currently focuses on **First-Order (FO)** operations, providing efficient single-level reformulations that avoid expensive Hessian computations.

| Library | Functional Role | Supported Atomic Operations |
| :--- | :--- | :--- |
| **FO-OL**<br>*(First-Order)* | **Constructs single-level surrogates.**<br>Reformulates the nested problem into first-order objectives using value-functions or penalties. | • **[VFO](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6dddcff5b115b40c998a08fbd1cea4d7-Abstract-Conference.html)** (Value-Function First-Order / BOME)<br>• **[VSO](http://proceedings.mlr.press/v139/liu21o.html)** (Value-Function Sequential)<br>• **[MESO](https://arxiv.org/abs/2405.09927)** (Moreau Envelope)<br>• **[PGDO](https://proceedings.mlr.press/v202/shen23c.html)** (Penalty Gradient Descent) |

</div>

## 🔨 Installation

BOAT-ms is built on top of **MindSpore**. Please install a suitable version for your hardware (Ascend / GPU / CPU) first.

### 1. Install MindSpore
Follow the [Official Installation Guide](https://www.mindspore.cn/install) or use the reference commands below:

**CPU (Linux-x86_64) Example**
```bash
# 1. Create environment
conda create -n mindspore_py39 python=3.9.11 -y
conda activate mindspore_py39

# 2. Install GCC dependencies (Linux only)
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-9 -y

# 3. Install MindSpore (Adjust MS_VERSION as needed)
export MS_VERSION=2.4.1
pip install mindspore==${MS_VERSION} -i [https://repo.mindspore.cn/pypi/simple](https://repo.mindspore.cn/pypi/simple) --trusted-host repo.mindspore.cn
```

### 2. Install BOAT-ms
Once MindSpore is ready, install BOAT-ms via PyPI or Source:
```bash
# Install from PyPI
pip install boat-ms

# Or install from Source (Specific Branch)
git clone -b boat_ms --single-branch [https://github.com/callous-youth/BOAT.git](https://github.com/callous-youth/BOAT.git)
cd BOAT
pip install -e .
```

## ⚡ How to Use BOAT

BOAT separates the **problem definition** from the **solver configuration**. Below is a MindSpore-based example.

### 1. Load Configurations
```python
import json
import boat_ms as boat

# Load algorithmic configurations
with open("configs/boat_config.json", "r") as f:
    boat_config = json.load(f)

# Load objective configurations
with open("configs/loss_config.json", "r") as f:
    loss_config = json.load(f)
```
### 2. Define Models and Optimizers
Use standard MindSpore models and optimizers.

```python
import mindspore as ms

# Define models
upper_model = MyUpperModel()
lower_model = MyLowerModel()

# Define optimizers (MindSpore syntax)
upper_opt = ms.nn.Adam(upper_model.trainable_params(), learning_rate=1e-3)
lower_opt = ms.nn.SGD(lower_model.trainable_params(), learning_rate=1e-2)
```
### 3. Customize & Initialize Problem
Specify the First-Order operation (e.g., VFO) and inject runtime objects.
```python
# Configure BOAT with MindSpore models/optimizers
boat_config["fo_ol"] = ["VFO"]  # Select First-Order Operation
boat_config["lower_level_model"] = lower_model
boat_config["upper_level_model"] = upper_model
boat_config["lower_level_opt"] = lower_opt
boat_config["upper_level_opt"] = upper_opt

# Initialize the BOAT core
b_optimizer = boat.Problem(boat_config, loss_config)
```

### 4. Build Solvers
Automatically construct the computation graph for the bilevel strategy.
```python
# Pass optimizers explicitly if needed by the solver backend
b_optimizer.build_ll_solver(lower_opt)
b_optimizer.build_ul_solver(upper_opt)
```

### 5. Run Optimization Loop
```python
# Training loop
for x_itr in range(1000):
    # Prepare data (MindSpore Tensors or dicts)
    ul_feed_dict = {"data": ul_data, "target": ul_target}
    ll_feed_dict = {"data": ll_data, "target": ll_target}
    
    # Run one iteration
    loss, run_time = b_optimizer.run_iter(ll_feed_dict, ul_feed_dict, current_iter=x_itr)
    
    if x_itr % 100 == 0:
        print(f"Iter {x_itr}: UL Loss {loss:.4f}")
```

## 🌍 Applications

BOAT covers a wide spectrum of BLO applications, categorized by the optimization target:

* **Data-Centric**: Data Hyper-Cleaning, Synthetic Data Reweighting, Diffusion Model Guidance.
* **Model-Centric**: Neural Architecture Search (NAS), LLM Prompt Optimization, Parameter Efficient Fine-Tuning (PEFT).
* **Strategy-Centric**: Meta-Learning, Hyperparameter Optimization (HO), Reinforcement Learning from Human Feedback (RLHF).

## 📝 Citation

If you find BOAT useful in your research, please consider citing our paper:

```bibtex
@article{liu2025boat,
  title={BOAT: A Compositional Operation Toolbox for Gradient-based Bi-Level Optimization},
  author={Liu, Yaohua and Pan, Jibao and Jiao, Xianghao and Gao, Jiaxin and Liu, Zhu and Liu, Risheng},
  journal={Submitted to Journal of Machine Learning Research (JMLR)},
  year={2025}
}
```

## License

MIT License

Copyright (c) 2024 Yaohua Liu

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
