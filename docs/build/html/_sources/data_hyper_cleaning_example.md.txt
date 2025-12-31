# Data HyperCleaning

This example demonstrates how to use the BOAT library to perform bi-level optimization with data hyper-cleaning.

---

## Step 1: Data Preparation

```python
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import boat_torch as boat
import torch
from util_file import data_splitting, initialize
from torchvision.datasets import MNIST

base_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(base_folder)
dataset = MNIST(root=os.path.join(parent_folder, "data/"), train=True, download=True)
tr, val, test = data_splitting(dataset, 5000, 5000, 10000)
tr.data_polluting(0.5)
tr.data_flatten()
val.data_flatten()
test.data_flatten()
```

### Explanation:
- The `MNIST` dataset is loaded from the specified directory.
- The `data_splitting` function splits the dataset into 5000 training, 5000 validation, and 10000 test samples.
- The `data_polluting` function introduces noise into the training data by randomly changing 50% of the values.
- The `data_flatten` function flattens the data to make it suitable for feeding into the model.

---

## Step 2: Model Definition

```python
class Net_x(torch.nn.Module):
    def __init__(self, tr):
        super(Net_x, self).__init__()
        self.x = torch.nn.Parameter(
            torch.zeros(tr.data.shape[0]).to(device).requires_grad_(True)
        )

    def forward(self, y):
        y = torch.sigmoid(self.x) * y
        y = y.mean()
        return y

x = Net_x(tr)
y = torch.nn.Sequential(torch.nn.Linear(28**2, 10)).to(device)
```

### Explanation:
- **`Net_x`**: A custom PyTorch model with a learnable parameter `x`. This parameter will be optimized as part of the lower-level optimization process.
- **`y` model**: A simple neural network with a single linear layer.

---

## Step 3: Optimizer and Initialization

```python
x_opt = torch.optim.Adam(x.parameters(), lr=0.01)
y_opt = torch.optim.SGD(y.parameters(), lr=0.01)
initialize(x)
initialize(y)
```

### Explanation:
- **Optimizers**: Adam optimizer is used for the lower-level model (`x`), and SGD is used for the upper-level model (`y`).
- **Initialization**: The `initialize` function resets the model parameters before training.

---

## Step 4: Configuration Loading

```python
with open(os.path.join(parent_folder, "data_hyper_cleaning/configs/boat_config_dhl.json"), "r") as f:
    boat_config = json.load(f)

with open(os.path.join(parent_folder, "data_hyper_cleaning/configs/loss_config_dhl.json"), "r") as f:
    loss_config = json.load(f)
```

### Explanation:
- Configuration files for BOAT are loaded, including:
  - **`boat_config`**: Contains configuration for the optimization process.
  - **`loss_config`**: Defines the loss functions used for training.
---

## Step 5: Main Function

```python

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Data HyperCleaner")

    parser.add_argument(
        "--gm_op",
        type=str,
        default="NGD",
        help="omniglot or miniimagenet or tieredImagenet",
    )
    parser.add_argument(
        "--na_op",
        type=str,
        default="RAD",
        help="convnet for 4 convs or resnet for Residual blocks",
    )
    parser.add_argument(
        "--fo_op",
        type=str,
        default=None,
        help="convnet for 4 convs or resnet for Residual blocks",
    )

    args = parser.parse_args()
    gm_op = args.gm_op.split(",") if args.gm_op else None
    na_op = args.na_op.split(",") if args.na_op else None
    boat_config["gm_op"] = gm_op
    boat_config["na_op"] = na_op
    boat_config["fo_op"] = args.fo_op
    boat_config["lower_level_model"] = y
    boat_config["upper_level_model"] = x
    boat_config["lower_level_opt"] = y_opt
    boat_config["upper_level_opt"] = x_opt
    boat_config["lower_level_var"] = list(y.parameters())
    boat_config["upper_level_var"] = list(x.parameters())
    b_optimizer = boat.Problem(boat_config, loss_config)
    b_optimizer.build_ll_solver()
    b_optimizer.build_ul_solver()
    ul_feed_dict = {"data": val.data.to(device), "target": val.clean_target.to(device)}
    ll_feed_dict = {"data": tr.data.to(device), "target": tr.dirty_target.to(device)}
    iterations = 3
    for x_itr in range(iterations):
        b_optimizer.run_iter(ll_feed_dict, ul_feed_dict, current_iter=x_itr)
```

### Explanation:
1. **Argument Parsing**:
   - `gm_op`: Specifies the list of the gradient mapping operations, e.g., ["NGD","GDA"].
   - `na_op`: Specifies the list of numerical approximation operations, e.g., ["RAD","RGT"].
   - `fo_op`: Optionally specifies a first-order gradient method, e.g., “MESO”.

2. **BOAT Configuration**:
   - Updates the `boat_config` with the parsed arguments and model components.
   - Initializes the BOAT `Problem` class for optimization.

3. **Iterative Optimization**:
   - Runs the optimization process for a specified number of iterations (`iterations`).
   - Computes and prints loss and runtime for each iteration.
