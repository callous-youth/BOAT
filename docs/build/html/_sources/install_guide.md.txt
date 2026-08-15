# Installation and Usage Guide

## 🔨 Installation

BOAT-ms is built on top of **MindSpore**. Please install a suitable version for your hardware (Ascend / GPU / CPU) first.

### 1. Install MindSpore
Follow the [Official Installation Guide](https://www.mindspore.cn/install) or use the reference commands below:

```bash
# 1. Create environment
conda create -n mindspore_py39 python=3.9.11 -y
conda activate mindspore_py39

# 2. Install MindSpore
pip install mindspore-dev
```

### 2. Install BOAT-ms
Once MindSpore is ready, install BOAT-ms via PyPI or Source:
```bash
# Install from PyPI
pip install boat-ms

# Or install from Source (Specific Branch)
git clone -b boat_ms --single-branch https://github.com/callous-youth/BOAT.git
cd BOAT
pip install -e .
```

## ✅ Quick Verification

Run the following command from the root directory of the BOAT repository:

```bash
python examples/L2_Reg_ms/l2_regularization.py
```

##  ⚡ **How to Use BOAT**

### **1. Load Configuration Files**
BOAT relies on two key configuration files:
- `boat_config.json`: Specifies optimization strategies and gradient mapping/numerical approximation operations.
- `loss_config.json`: Defines the loss functions for both levels of the BLO process.  
```python
import os
import json
import boat_ms as boat

# Load configuration files
with open("path_to_configs/boat_config.json", "r") as f:
    boat_config = json.load(f)

with open("path_to_configs/loss_config.json", "r") as f:
    loss_config = json.load(f)
```

### **2. Define Models and Optimizers**
You need to specify both the upper-level and lower-level models along with their respective optimizers.

```python
import mindspore 

# Define models
upper_model = UpperModel(*args, **kwargs)  # Replace with your upper-level model
lower_model = LowerModel(*args, **kwargs)  # Replace with your lower-level model

# Define optimizers
upper_opt = mindspore.nn.Adam(upper_model.parameters(), lr=0.01)
lower_opt = mindspore.optim.SGD(lower_model.parameters(), lr=0.01)
```

### **3. Customize BOAT Configuration**
Modify the boat_config to include your gradient mapping and numerical approximation methods, as well as model and variable details.

```python
# Example gradient mapping and numerical approximation methods Combination.
boat_config["fo_op"] = "MESO" # FOGO supports only

# Add methods and model details to the configuration
boat_config["lower_level_model"] = lower_model
boat_config["upper_level_model"] = upper_model
boat_config["lower_level_opt"] = lower_opt
boat_config["upper_level_opt"] = upper_opt
boat_config["lower_level_var"] = lower_model.trainable_params()
boat_config["upper_level_var"] = upper_model.trainable_params()
```

### **4. Initialize the BOAT Problem**
Modify the boat_config to include your gradient mapping and numerical approximation methods, as well as model and variable details.

```python
# Initialize the problem
b_optimizer = boat.Problem(boat_config, loss_config)

# Build solvers for lower and upper levels
b_optimizer.build_ll_solver()  # Lower-level solver
b_optimizer.build_ul_solver()  # Upper-level solver
```

### **5. Define Data Feeds**
Prepare the data feeds for both levels of the BLO process, which was further fed into the the upper-level  and lower-level objective functions. 

```python
# Define data feeds (Demo Only)
ul_feed_dict = {"data": upper_level_data, "target": upper_level_target}
ll_feed_dict = {"data": lower_level_data, "target": lower_level_target}
```

### **6. Run the Optimization Loop**
Execute the optimization loop, optionally customizing the solver strategy for gradient mapping methods.

```python
# Set number of iterations
iterations = 1000

# Optimization loop (Demo Only)
for x_itr in range(iterations):
    # Run a single optimization iteration
    loss, run_time = b_optimizer.run_iter(ll_feed_dict, ul_feed_dict, current_iter=x_itr)

```
