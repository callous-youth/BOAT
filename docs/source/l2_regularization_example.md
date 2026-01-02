# L2 Regularization with Jittor

This runnable example shows how to use the BOAT library with the Jittor backend to solve a bi-level optimization problem with L2 regularization, covering end-to-end data loading (sparse-to-dense conversion), model/optimizer setup, solver construction, and iterative training with evaluation.

## Step-by-Step Explanation

## Step 1: Imports & Path Setup

```python
import argparse
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jittor as jit
import boat_jit as boat

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups_vectorized

import json

```
### Explanation:
- Imports Jittor, BOAT-JIT, and scikit-learn utilities for dataset loading and splitting.


## Step 2: Configuration Loading

```python
base_folder = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_folder, "configs_jit/boat_config_l2.json"), "r") as f:
    boat_config = json.load(f)

with open(os.path.join(base_folder, "configs_jit/loss_config_l2.json"), "r") as f:
    loss_config = json.load(f)
```

### Explanation:
- **`boat_config_l2.json`**: Contains configuration for the bi-level optimization problem.
- **`loss_config_l2.json`**: Defines the loss functions for both upper-level and lower-level models.



## Step 3: Data Preparation

```python
def get_data(args, max_samples=2000):
    """
    Load and process data for Jittor, with optional downsampling.
    """
    def from_sparse(x):
        x = x.tocoo()
        values = x.data
        indices = np.vstack((x.row, x.col))
        i = jit.array(indices, dtype=jit.int64)
        v = jit.array(values, dtype=jit.float32)
        shape = x.shape
        dense_tensor = jit.zeros(shape, dtype=jit.float32)
        dense_tensor[i[0], i[1]] = v
        return dense_tensor

    val_size = 0.5

    train_x, train_y = fetch_20newsgroups_vectorized(
        subset="train",
        return_X_y=True,
        data_home=args.data_path,
        download_if_missing=True,
    )

    test_x, test_y = fetch_20newsgroups_vectorized(
        subset="test",
        return_X_y=True,
        data_home=args.data_path,
        download_if_missing=True,
    )

    # ---- New: subsampling to reduce dataset size ----
    if max_samples is not None:
        train_x = train_x[:max_samples]
        train_y = train_y[:max_samples]
        test_x = test_x[: max_samples // 2]
        test_y = test_y[: max_samples // 2]

    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, stratify=train_y, test_size=val_size
    )
    test_x, teval_x, test_y, teval_y = train_test_split(
        test_x, test_y, stratify=test_y, test_size=0.5
    )

    train_x, val_x, test_x, teval_x = map(from_sparse, [train_x, val_x, test_x, teval_x])
    train_y, val_y, test_y, teval_y = map(
        lambda y: jit.array(y, dtype=jit.int64), [train_y, val_y, test_y, teval_y]
    )

    print(train_y.shape[0], val_y.shape[0], test_y.shape[0], teval_y.shape[0])
    return (train_x, train_y), (val_x, val_y), (test_x, test_y), (teval_x, teval_y)

```

### Explanation:
- The `get_data` function loads the dataset, processes it to Jittor tensors, and splits it into training, validation, test, and evaluation sets.
- Processed data is saved to a file for future use.

## Step 4: Evaluation Helper
```python
def evaluate(x, w, testset):
    """
    Evaluate the performance of the model on the test set.

    Args:
        x (jittor.Var): Model weights (used in matrix multiply).
        w (jittor.Var): Upper-level variables (kept for interface consistency).
        testset (tuple): Tuple containing test_x and test_y.

    Returns:
        tuple: Loss and accuracy of the model on the test set.
    """
    with jit.no_grad():
        test_x, test_y = testset

        # logits
        y = test_x @ x

        # to numpy for accuracy
        y_np = y.numpy()
        test_y_np = test_y.numpy() if isinstance(test_y, jit.Var) else test_y

        loss = jit.nn.cross_entropy_loss(y, jit.array(test_y_np)).item()
        predicted = y_np.argmax(axis=-1)
        acc = (predicted == test_y_np).sum() / len(test_y_np)

    return loss, acc

```
### Explanation:
- Computes cross-entropy loss using Jittor.
- Computes accuracy using NumPy for simplicity.

## Step 5: Main Function & Argument Parsing

```python
def main():
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--generate_data", action="store_true", default=False)
        parser.add_argument("--pretrain", action="store_true", default=False)
        parser.add_argument("--epochs", type=int, default=1000)
        parser.add_argument("--iterations", type=int, default=10, help="T")
        parser.add_argument("--data_path", default="./data", help="where to save data")
        parser.add_argument("--model_path", default="./save_l2reg", help="where to save model")
        parser.add_argument("--x_lr", type=float, default=100)
        parser.add_argument("--xhat_lr", type=float, default=100)
        parser.add_argument("--w_lr", type=float, default=1000)
        parser.add_argument("--w_momentum", type=float, default=0.9)
        parser.add_argument("--x_momentum", type=float, default=0.9)
        parser.add_argument("--K", type=int, default=10, help="k")
        parser.add_argument("--u1", type=float, default=1.0)
        parser.add_argument("--BVFSM_decay", type=str, default="log", choices=["log", "power2"])
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument(
            "--alg",
            type=str,
            default="BOME",
            choices=["BOME", "BSG_1", "penalty", "AID_CG", "AID_FP", "ITD", "BVFSM",
                     "baseline", "VRBO", "reverse", "stocBiO", "MRBO"],
        )
        parser.add_argument("--gm_op", type=str, default="DM,NGD")
        parser.add_argument("--na_op", type=str, default="RAD")
        parser.add_argument("--fo_op", type=str, default=None)
        args = parser.parse_args()

        np.random.seed(args.seed)
        jit.set_global_seed(args.seed)
        return args

    args = parse_args()

```
### Explanation:
- Defines CLI arguments controlling dataset path, strategies (`gm_op`, `na_op`) or `fo_op`, and seeds.


## Step 6: Data Setting and Model Initialization

```python
trainset, valset, testset, tevalset = get_data(args)

jit.save(
    (trainset, valset, testset, tevalset),
    os.path.join(args.data_path, "l2reg.pkl")
)
print(f"[info] successfully generated data to {args.data_path}/l2reg.pkl")

class UpperModel(jit.Module):
    def __init__(self, n_feats):
        # Initialize learnable regularization parameters
        self.x = jit.init.constant([n_feats], "float32", 0.0).clone()

    def execute(self):
        return self.x

class LowerModel(jit.Module):
    def __init__(self, n_feats, num_classes):
        # Initialize classifier weights
        self.y = jit.zeros([n_feats, num_classes])
        jit.init.kaiming_normal_(
            self.y, a=0, mode="fan_in", nonlinearity="leaky_relu"
        )

    def execute(self):
        return self.y

upper_model = UpperModel(trainset[0].shape[-1])
lower_model = LowerModel(trainset[0].shape[-1], int(trainset[1].max().item()) + 1)
```

### Explanation:
- Saves the processed dataset for reuse as `l2reg.pkl`.
- **`UpperModel`**: Represents the upper-level model with a single learnable parameter.
- **`LowerModel`**: Represents the lower-level model initialized using the Kaiming initialization strategy.



## Step 7: Optimizer & Strategy Setup

```python
upper_opt = jit.nn.Adam(upper_model.parameters(), lr=0.01)
lower_opt = jit.nn.SGD(lower_model.parameters(), lr=0.01)

# Parse optimization strategies from arguments
gm_op = args.gm_op.split(",") if args.gm_op else None
na_op = args.na_op.split(",") if args.na_op else None
```

### Explanation:
- **Adam optimizer**: Used for the upper-level model to update its parameters.
- **SGD optimizer**: Applied to the lower-level model for efficient gradient updates.
- The `gm_op` and `na_op` parameters allow flexible optimization strategies.



## Step 8: Bi-Level Optimization Setup

```python
# Configure BOAT problem
if na_op is not None:
    if "RGT" in na_op:
        boat_config["RGT"]["truncate_iter"] = 1
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

### Explanation:
- Configures the `boat_config` with models, optimizers, and variables for both levels.
- Instantiates the `boat.Problem` class and builds the necessary lower-level and upper-level solvers.

---

## Step 9: Optimization Loop

```python
ul_feed_dict = {"data": trainset[0], "target": trainset[1]}
ll_feed_dict = {"data": valset[0], "target": valset[1]}

# Determine iteration count based on strategy
if "DM" in boat_config["gm_op"] and ("GDA" in boat_config["gm_op"]):
    iterations = 3
else:
    iterations = 2

for x_itr in range(iterations):
    # Dynamic strategy adjustment for Dynamic Methods (DM)
    if "DM" in boat_config["gm_op"] and boat_config["fo_op"] is None:
        if "GDA" in boat_config["gm_op"]:
             b_optimizer._ll_solver.gradient_instances[-1].strategy = "s" + str(x_itr % 3 + 1)
        else:
             b_optimizer._ll_solver.gradient_instances[-1].strategy = "s" + str(1)

    loss, run_time = b_optimizer.run_iter(
        ll_feed_dict, ul_feed_dict, current_iter=x_itr
    )
```

### Explanation:
- The `evaluate` function calculates the model's loss and accuracy on the test dataset.
- Outputs the test performance metrics for monitoring optimization progress.
- The `run_iter` function iterates over the bi-level optimization process using BOAT.



---

## Step 10: Entry Point

```python
if __name__ == "__main__":
    main()

```

### Explanation:
- Standard Python entry point that makes the script runnable directly.

## How to Run

To execute the example, use the following command:

```bash
python your_script_name.py --data_path ./data --model_path ./save_l2reg --gm_op NGD --na_op RAD
