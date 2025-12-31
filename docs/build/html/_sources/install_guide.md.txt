# Installation and Usage Guide

## 🔨 Installation

BOAT-jit is built on top of **Jittor**. Please ensure Jittor is installed correctly before installing BOAT.

### 1. Install Jittor
Follow the [Official Installation Guide](https://cg.cs.tsinghua.edu.cn/jittor/download/) or use the commands below:

**Linux (Ubuntu / CentOS)**

```bash
sudo apt install python3.8-dev libomp-dev
python3.8 -m pip install jittor
# Verify installation (Optional)
python3.8 -m jittor.test.test_example
```

**Windows**

```bash
python -m pip install jittor
python -m jittor.test.test_core
```

**macOS**

```bash
brew install libomp
python3.8 -m pip install jittor
```

### 2. Install BOAT-jit
Once Jittor is ready, install BOAT-jit via PyPI or Source:
```bash
# Install from PyPI
pip install boat-jit

# Or install from Source (Specific Branch)
git clone -b boat_jit --single-branch [https://github.com/callous-youth/BOAT.git](https://github.com/callous-youth/BOAT.git)
cd BOAT
pip install -e .
```

##  ⚡ **How to Use BOAT**

### **1. Load Configuration Files**
BOAT relies on two key configuration files:
- `boat_config.json`: Specifies optimization strategies and dynamic/hyper-gradient operations.
- `loss_config.json`: Defines the loss functions for both levels of the BLO process.  
```python
import os
import json
import boat_jit as boat

# Load configuration files
with open("path_to_configs/boat_config.json", "r") as f:
    boat_config = json.load(f)

with open("path_to_configs/loss_config.json", "r") as f:
    loss_config = json.load(f)
```

### **2. Define Models and Optimizers**
You need to specify both the upper-level and lower-level models along with their respective optimizers.

```python
import torch

# Define models
upper_model = UpperModel(*args, **kwargs)  # Replace with your upper-level model
lower_model = LowerModel(*args, **kwargs)  # Replace with your lower-level model

# Define optimizers
upper_opt = jit.nn.Adam(upper_model.parameters(), lr=0.01)
lower_opt = jit.nn.SGD(lower_model.parameters(), lr=0.01)
```

### **3. Customize BOAT Configuration**
Modify the boat_config to include your gradient mapping and numerical approximation operation, as well as model and variable details.

```python
# Example gradient mapping and numerical approximation operation Combination.
gm_op = ["NGD", "DI", "GDA"]  # Gradient Mapping Operations (Demo Only)
na_op = ["RGT","RAD"]          # Numerical Approximation Operations (Demo Only)

# Add methods and model details to the configuration
boat_config["gm_op"] = gm_op
boat_config["na_op"] = na_op
boat_config["lower_level_model"] = lower_model
boat_config["upper_level_model"] = upper_model
boat_config["lower_level_opt"] = lower_opt
boat_config["upper_level_opt"] = upper_opt
boat_config["lower_level_var"] = list(lower_model.parameters())
boat_config["upper_level_var"] = list(upper_model.parameters())
```

### **4. Initialize the BOAT Problem**
Modify the boat_config to include your gradient mapping and numerical approximation operation, as well as model and variable details.

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
Execute the optimization loop, optionally customizing the solver strategy for gradient mapping operations.

```python
# Set number of iterations
iterations = 1000

# Optimization loop (Demo Only)
for x_itr in range(iterations):
    # Run a single optimization iteration
    loss, run_time = b_optimizer.run_iter(ll_feed_dict, ul_feed_dict, current_iter=x_itr)

```
