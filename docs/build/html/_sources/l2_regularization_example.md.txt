# L2 Regularization

This example demonstrates how to use the BOAT library to perform bi-level optimization with L2 regularization. The example includes data preprocessing, model initialization, and the optimization process.

---

## Step 1: Imports and Environment Setup

```python
import argparse
import numpy as np
import torch
import boat_torch as boat
import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from examples.L2_Reg.utils_l2 import get_data, UpperModel, LowerModel
```

### Explanation:
- Core libraries such as **`torch`**, **`numpy`**, and **`boat_torch`** are imported.

- The project root directory is appended to  **`sys.path`** to allow importing local utility modules.

- Helper functions and model definitions for L2 regularization are imported from **`utils_l2`**.

---

## Step 2: Configuration Loading

```python
base_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(base_folder)

with open(os.path.join(parent_folder, "L2_Reg/configs/boat_config_l2.json"), "r") as f:
    boat_config = json.load(f)

with open(os.path.join(parent_folder, "L2_Reg/configs/loss_config_l2.json"), "r") as f:
    loss_config = json.load(f)

```

### Explanation:
- BOAT optimization settings are loaded from `boat_config_l2.json`.
- Loss functions for upper- and lower-level optimization are defined in `loss_config_l2.json`.

---

## Step 3: Argument Parsing and Reproducibility

```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate_data",
        action="store_true",
        default=False,
        help="whether to create data",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        default=False,
        help="whether to create data",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--iterations", type=int, default=10, help="T")
    parser.add_argument("--data_path", default="./data", help="where to save data")
    parser.add_argument(
        "--model_path", default="./save_l2reg", help="where to save model"
    )
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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return args
```

### Explanation:
- Command-line arguments control dataset generation, optimization iterations, and BOAT operators.
- Random seeds are fixed to ensure reproducible experimental results.
- `gm_op`, `na_op`, and `fo_op` define the bilevel optimization strategy used by BOAT.

---
## Step 4: Data Preparation

```python
args = parse_args()
trainset, valset, testset, tevalset = get_data(args)

torch.save(
    (trainset, valset, testset, tevalset),
    os.path.join(args.data_path, "l2reg.pt")
)
```

### Explanation:
- Training, validation, test, and evaluation datasets are generated using `get_data`.
- The processed datasets are saved locally for later reuse.
- 
---

## Step 5: Model Definition

```python
device = torch.device("cpu")
n_feats = trainset[0].shape[-1]

upper_model = UpperModel(n_feats, device)
lower_model = LowerModel(
    n_feats, device, num_classes=trainset[1].unique().shape[-1]
)

```

### Explanation:
- **`UpperModel`**: Represents the upper-level model, optimizing high-level objectives.
- **`LowerModel`**: Represents the lower-level model, focusing on optimizing low-level objectives.

---
## Step 6: Optimizer Setup

```python
upper_opt = torch.optim.Adam(upper_model.parameters(), lr=0.01)
lower_opt = torch.optim.SGD(lower_model.parameters(), lr=0.01)
```
---
## Step 7: BOAT Problem Construction

```python
gm_op = args.gm_op.split(",") if args.gm_op else []
na_op = args.na_op.split(",") if args.na_op else []

boat_config["gm_op"] = gm_op
boat_config["na_op"] = na_op
boat_config["fo_op"] = args.fo_op
boat_config["lower_level_model"] = lower_model
boat_config["upper_level_model"] = upper_model
boat_config["lower_level_opt"] = lower_opt
boat_config["upper_level_opt"] = upper_opt
boat_config["lower_level_var"] = list(lower_model.parameters())
boat_config["upper_level_var"] = list(upper_model.parameters())

b_optimizer = boat.Problem(boat_config, loss_config)
b_optimizer.build_ll_solver()
b_optimizer.build_ul_solver()

```

- BOAT configuration is updated with models, optimizers, and variables for both levels.
- The `Problem` class initializes the bilevel optimization framework.


---
## Step 8: BOAT Problem Construction

```python
ul_feed_dict = {"data": trainset[0].to(device), "target": trainset[1].to(device)}
ll_feed_dict = {"data": valset[0].to(device), "target": valset[1].to(device)}

iterations = 30
for x_itr in range(iterations):
    b_optimizer.run_iter(ll_feed_dict, ul_feed_dict, current_iter=x_itr)

```