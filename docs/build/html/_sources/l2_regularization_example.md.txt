# L2 Regularization with MindSpore

This example demonstrates how to use the BOAT library with MindSpore to perform bi-level optimization with L2 regularization. The example includes data preprocessing, model initialization, and the optimization process.

---

## Step 1: Configuration Loading

```python
with open(os.path.join(base_folder, "configs_ms/boat_config_l2.json"), "r") as f:
    boat_config = json.load(f)

with open(os.path.join(base_folder, "configs_ms/loss_config_l2.json"), "r") as f:
    loss_config = json.load(f)
```

### Explanation:
- **`boat_config_l2.json`**: Contains configuration for the bi-level optimization problem.
- **`loss_config_l2.json`**: Defines the loss functions for both upper-level and lower-level models.

---

## Step 2: Data Preparation

```python
trainset, valset, testset, tevalset = get_data(args)
save_path = os.path.join(args.data_path, "l2reg.pkl")
save_data((trainset, valset, testset, tevalset), save_path)
```

### Explanation:
- The `get_data` function loads and splits the dataset into training, validation, testing, and evaluation sets.
- Processed data is saved to the specified `data_path` directory using a custom `save_data` function that supports MindSpore COOTensors.

---

## Step 3: Model Initialization

```python
n_feats = trainset[0].shape[-1]
num_classes = int(trainset[1].max().asnumpy()) + 1

class UpperModel(nn.Cell):
    def __init__(self, n_feats):
        super(UpperModel, self).__init__()
        self.x = ms.Parameter(ms.Tensor(np.zeros(n_feats), ms.float32))

    def construct(self):
        return self.x

class LowerModel(nn.Cell):
    def __init__(self, n_feats, num_classes):
        super(LowerModel, self).__init__()
        he_normal = HeNormal()
        self.y = ms.Parameter(
            ms.Tensor(shape=(n_feats, num_classes), dtype=ms.float32, init=he_normal)
        )

    def construct(self):
        return self.y

upper_model = UpperModel(n_feats)
lower_model = LowerModel(n_feats, num_classes)
```

### Explanation:
- **`UpperModel`**: Represents the upper-level model, with trainable parameters initialized to zeros.
- **`LowerModel`**: Represents the lower-level model, with trainable weights initialized using HeNormal.

---

## Step 4: Optimizer Setup

```python
upper_opt = nn.Adam(upper_model.trainable_params(), learning_rate=0.1)
lower_opt = nn.SGD(lower_model.trainable_params(), learning_rate=0.1)
```

### Explanation:
- **Adam optimizer** is used for the upper-level model.
- **SGD optimizer** is applied to the lower-level model.

---

## Step 5: BOAT Problem Setup

```python
boat_config["lower_level_model"] = lower_model
boat_config["upper_level_model"] = upper_model
boat_config["lower_level_opt"] = lower_opt
boat_config["upper_level_opt"] = upper_opt
boat_config["lower_level_var"] = lower_model.trainable_params()
boat_config["upper_level_var"] = upper_model.trainable_params()

b_optimizer = boat.Problem(boat_config, loss_config)
b_optimizer.build_ll_solver()
b_optimizer.build_ul_solver()
```

### Explanation:
- Configures the BOAT library with the models, optimizers, and parameters.
- Initializes the lower-level and upper-level solvers for bi-level optimization.

---

## Step 6: Training Loop

```python
ul_feed_dict = {"data": trainset[0], "target": trainset[1]}
ll_feed_dict = {"data": valset[0], "target": valset[1]}

iterations = 30
for x_itr in range(iterations):
    loss, run_time = b_optimizer.run_iter(
        ll_feed_dict, ul_feed_dict, current_iter=x_itr
    )

    if x_itr % 1 == 0:
        test_loss, test_acc = evaluate(lower_model(), upper_model(), testset)
        teval_loss, teval_acc = evaluate(lower_model(), upper_model(), tevalset)
        print(
            f"[info] epoch {x_itr:5d} te loss {test_loss:10.4f} te acc {test_acc:10.4f} teval loss {teval_loss:10.4f} teval acc {teval_acc:10.4f} time {run_time:8.2f}"
        )
```

### Explanation:
- **Input feed dictionaries**: Define data and targets for both lower-level and upper-level optimization.
- **`run_iter`**: Performs one iteration of bi-level optimization.
- Evaluation is performed every epoch, and results are printed.

---

## Step 7: Evaluation

```python
def evaluate(x, w, testset):
    test_x, test_y = testset

    if isinstance(test_x, ms.COOTensor):
        y = ops.SparseTensorDenseMatmul()(test_x.indices, test_x.values, test_x.shape, x)
    else:
        y = ops.MatMul()(test_x, x)

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    loss = loss_fn(y, test_y).mean()
    loss = loss.asnumpy().item()

    predictions = y.argmax(axis=-1)
    acc = (predictions == test_y).sum().asnumpy() / test_y.shape[0]

    return loss, acc
```

### Explanation:
- Evaluates the model on the test set, computing loss and accuracy.
- Supports dense and sparse tensors (COOTensor) for input data.

---

## Running the Example

To execute the script, use the following command:

```bash
python your_script_name.py --data_path ./data --iterations 30
```

### Key Features:
- Configurable via command-line arguments.
- Supports saving and loading custom data formats with COOTensor for efficient processing.
- Demonstrates bi-level optimization with MindSpore and the BOAT library.
