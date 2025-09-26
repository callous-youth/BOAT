# Installation and Usage Guide

##  🔨 **Installation**
To install BOAT with *PyPi*, use the following command:
```bash
pip install boat-jit
```
or you can install the latest version from the source code on *GitHub*:
```bash
git clone -b boat_jit --single-branch https://github.com/callous-youth/BOAT.git

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
Modify the boat_config to include your dynamic and hyper-gradient methods, as well as model and variable details.

```python
# Example dynamic and hyper-gradient methods Combination.
dynamic_method = ["NGD", "DI", "GDA"]  # Dynamic Methods (Demo Only)
hyper_method = ["RGT","RAD"]          # Hyper-Gradient Methods (Demo Only)

# Add methods and model details to the configuration
boat_config["dynamic_op"] = dynamic_method
boat_config["hyper_op"] = hyper_method
boat_config["lower_level_model"] = lower_model
boat_config["upper_level_model"] = upper_model
boat_config["lower_level_opt"] = lower_opt
boat_config["upper_level_opt"] = upper_opt
boat_config["lower_level_var"] = list(lower_model.parameters())
boat_config["upper_level_var"] = list(upper_model.parameters())
```

### **4. Initialize the BOAT Problem**
Modify the boat_config to include your dynamic and hyper-gradient methods, as well as model and variable details.

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
Execute the optimization loop, optionally customizing the solver strategy for dynamic methods.

```python
# Set number of iterations
iterations = 1000

# Optimization loop (Demo Only)
for x_itr in range(iterations):
    # Run a single optimization iteration
    loss, run_time = b_optimizer.run_iter(ll_feed_dict, ul_feed_dict, current_iter=x_itr)

```
