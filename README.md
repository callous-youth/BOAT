
# BOAT - Problem Adaptive Operation Toolbox for Gradient-based Bilevel Optimization
[![PyPI version](https://badge.fury.io/py/boml.svg)](https://badge.fury.io/py/boml)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/callous-youth/BOAT/workflow.yml)
[![codecov](https://codecov.io/github/callous-youth/BOAT/graph/badge.svg?token=0MKAOQ9KL3)](https://codecov.io/github/callous-youth/BOAT)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/callous-youth/BOAT)
[![pages-build-deployment](https://github.com/callous-youth/BOAT/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/callous-youth/BOAT/actions/workflows/pages/pages-build-deployment)
![GitHub top language](https://img.shields.io/github/languages/top/callous-youth/BOAT)
![GitHub language count](https://img.shields.io/github/languages/count/callous-youth/BOAT)
![Python version](https://img.shields.io/pypi/pyversions/boml)
![license](https://img.shields.io/badge/license-MIT-000000.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)


**BOAT** (Bi-level Optimization Adaptive Toolbox) is a sophisticated, open-source Python library crafted to support flexible, efficient, and highly customizable gradient-based bi-level optimization (BLO). Designed with a modular structure, BOAT empowers researchers and developers to tackle a broad spectrum of BLO applications in machine learning and computer vision.

## 🔑  **Key Features**
- **Dynamic Operation Library (D-OL)**: Incorporates four advanced dynamic system construction operations, enabling users to flexibly tailor optimization trajectories for BLO tasks.
- **Hyper-Gradient Operation Library (H-OL)**: Provides nine refined operations for hyper-gradient computation, significantly enhancing the precision and efficiency of gradient-based BLO methods.
- **First-Order Gradient Methods (FOGMs)**: Integrates three state-of-the-art first-order methods, enabling fast prototyping and validation of new BLO algorithms.
- **Modularized Design for Customization**: Empowers users to flexibly combine and configure dynamic and hyper-gradient operations, facilitating the design of novel optimization algorithms.
- **Comprehensive Testing & Continuous Integration**: Ensures software reliability with 96% code coverage, functional tests with pytest, and continuous integration through GitHub Actions.
- **Detailed Documentation & Community Support**: Offers thorough documentation with practical examples and API references via MkDocs, ensuring accessibility and ease of use for both novice and advanced users.

##  🚀 **Why BOAT?**
Unlike conventional automatic differentiation (AD) tools that focus on specific optimization schemes (e.g., explicit or implicit methods), **BOAT** extends the landscape of BLO applications by supporting a broader range of problem-adaptive operations. It bridges the gap between theoretical research and practical deployment, offering extensive flexibility to design, customize, and accelerate BLO techniques.

##  🏭 **Applications**
BOAT enables efficient implementation and adaptation of advanced BLO techniques for key applications, including but not limited to:
- **Hyperparameter Optimization (HO)**
- **Neural Architecture Search (NAS)**
- **Adversarial Training (AT)**
- **Low-Light Image Enhancement (LLE)**
- **Multi-Modality Image Fusion (MIF)**
- **Medical Image Analysis (MIA)**

##  ⚡ **Technical Highlights**
- **Extensible Operation Library**: 4 dynamic system operations (D-OL) and 9 hyper-gradient operations (H-OL) for enhanced configurability.
- **Advanced First-Order Methods**: Supports Value-function Sequential Method (VSM), Value-function First-order Method (VFM), and Moreau Envelope Single-loop Method (MESM).
- **Unified Computational Analysis**: Offers a comprehensive complexity analysis of gradient-based BLO techniques to guide users in selecting optimal configurations for efficiency and accuracy.
- **Fast Prototyping & Algorithm Validation**: Streamlined support for defining, testing, and benchmarking new BLO algorithms.

##  🔨 **Installation**
To install BOAT, use the following command:
```bash
pip install boat
```


[comment]: <> (BOML is a modularized optimization library that unifies several ML algorithms into a common bilevel optimization framework. It provides interfaces to implement popular bilevel optimization algorithms, so that you could quickly build your own meta learning neural network and test its performance.)

[comment]: <> (ReadMe.md contains brief introduction to implement meta-initialization-based and meta-feature-based methods in few-shot classification field. Except for algorithms which have been proposed, various combinations of lower level and upper level strategies are available. )

[comment]: <> (## Meta Learning )

[comment]: <> (Meta learning works fairly well when facing incoming new tasks by learning an initialization with favorable generalization capability. And it also has good performance even provided with a small amount of training data available, which gives birth to various solutions for different application such as few-shot learning problem.)

[comment]: <> (We present a general bilevel optimization paradigm to unify different types of meta learning approaches, and the mathematical form could be summarized as below:<br>)

[comment]: <> (<div align=center>)
  
[comment]: <> (![Bilevel Optimization Model]&#40;https://github.com/dut-media-lab/BOML/blob/master/figures/p1.png&#41;)

[comment]: <> (</div>)

[comment]: <> (## Generic Optimization Routine)

[comment]: <> (Here we illustrate the generic optimization process and hierarchically built strategies in the figure, which could be quikcly implemented in the following example.<br>)

[comment]: <> (<div align=center>)
  
[comment]: <> (![Optimization Routine]&#40;https://github.com/dut-media-lab/BOML/blob/master/figures/p2.png&#41;)

[comment]: <> (</div>)

[comment]: <> (## Documentation )

[comment]: <> (For more detailed information of basic function and construction process, please refer to our [Documentation]&#40;https://boml.readthedocs.io&#41; or[Project Page]&#40;https://dut-media-lab.github.io/BOML/&#41;. Scripts in the directory named test_script are useful for constructing general training process.)

[comment]: <> (Here we give recommended settings for specific hyper paremeters to quickly test performance of popular algorithms.)

[comment]: <> (## Running examples)

[comment]: <> (### Start from loading data)

[comment]: <> (```python)

[comment]: <> (import boml)

[comment]: <> (from boml import utils)

[comment]: <> (from test_script.script_helper import *)

[comment]: <> (dataset = boml.load_data.meta_omniglot&#40;)

[comment]: <> (    std_num_classes=args.classes,)

[comment]: <> (    examples_train=args.examples_train,)

[comment]: <> (    examples_test=args.examples_test,)

[comment]: <> (&#41;)

[comment]: <> (# create instance of BOMLExperiment for ong single task)

[comment]: <> (ex = boml.BOMLExperiment&#40;dataset&#41;)

[comment]: <> (```)

[comment]: <> (### Build network structure and define parameters for meta-learner and base-learner)

[comment]: <> (```python)

[comment]: <> (boml_ho = boml.BOLVOptimizer&#40;)

[comment]: <> (    method="MetaInit", inner_method="Simple", outer_method="Simple")

[comment]: <> (&#41;)

[comment]: <> (meta_learner = boml_ho.meta_learner&#40;_input=ex.x, dataset=dataset, meta_model="V1"&#41;)

[comment]: <> (ex.adapt_model = boml_ho.base_learner&#40;_input=ex.x, meta_learner=meta_learner&#41;)

[comment]: <> (``` )

[comment]: <> (### Define LL objectives and LL calculation process)

[comment]: <> (```python)

[comment]: <> (loss_inner = utils.cross_entropy&#40;pred=ex.adapt_model.out, label=ex.y&#41;)

[comment]: <> (accuracy = utils.classification_acc&#40;pred=ex.adapt_model.out, label=ex.y&#41;)

[comment]: <> (inner_grad = boml_ho.ll_problem&#40;)

[comment]: <> (    inner_objective=loss_inner,)

[comment]: <> (    learning_rate=args.lr,)

[comment]: <> (    T=args.T,)

[comment]: <> (    experiment=ex,)

[comment]: <> (    var_list=ex.adapt_model.var_list,)

[comment]: <> (&#41;)

[comment]: <> (```)

[comment]: <> (### Define UL objectives and UL calculation process)

[comment]: <> (```python)

[comment]: <> (loss_outer = utils.cross_entropy&#40;pred=ex.adapt_model.re_forward&#40;ex.x_&#41;.out, label=ex.y_&#41;  # loss function)

[comment]: <> (boml_ho.ul_problem&#40;)

[comment]: <> (    outer_objective=loss_outer,)

[comment]: <> (    meta_learning_rate=args.meta_lr,)

[comment]: <> (    inner_grad=inner_grad,)

[comment]: <> (    meta_param=tf.get_collection&#40;boml.extension.GraphKeys.METAPARAMETERS&#41;,)

[comment]: <> (&#41;)

[comment]: <> (```)

[comment]: <> (### Aggregate all the defined operations)

[comment]: <> (```python)

[comment]: <> (# Only need to be called once after all the tasks are ready)

[comment]: <> (boml_ho.aggregate_all&#40;&#41;)

[comment]: <> (```)

[comment]: <> (### Meta training iteration)

[comment]: <> (```python)

[comment]: <> (with tf.Session&#40;&#41; as sess:)

[comment]: <> (    tf.global_variables_initializer&#40;&#41;.run&#40;session=sess&#41;)

[comment]: <> (    for itr in range&#40;args.meta_train_iterations&#41;:)

[comment]: <> (        # Generate the feed_dict for calling run&#40;&#41; everytime)

[comment]: <> (        train_batch = BatchQueueMock&#40;)

[comment]: <> (            dataset.train, 1, args.meta_batch_size, utils.get_rand_state&#40;1&#41;)

[comment]: <> (        &#41;)

[comment]: <> (        tr_fd, v_fd = utils.feed_dict&#40;train_batch.get_single_batch&#40;&#41;, ex&#41;)

[comment]: <> (        # Meta training step)

[comment]: <> (        boml_ho.run&#40;tr_fd, v_fd&#41;)

[comment]: <> (        if itr % 100 == 0:)

[comment]: <> (            print&#40;sess.run&#40;loss_inner, utils.merge_dicts&#40;tr_fd, v_fd&#41;&#41;&#41;)

[comment]: <> (```)

[comment]: <> (## Related Methods )

[comment]: <> ( - [Hyperparameter optimization with approximate gradient&#40;HOAG&#41;]&#40;https://arxiv.org/abs/1602.02355&#41;)

[comment]: <> ( - [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks&#40;MAML&#41;]&#40;https://arxiv.org/abs/1703.03400&#41;)

[comment]: <> ( - [On First-Order Meta-Learning Algorithms&#40;FMAML&#41;]&#40;https://arxiv.org/abs/1703.03400&#41;)

[comment]: <> ( - [Meta-SGD: Learning to Learn Quickly for Few-Shot Learning&#40;Meta-SGD&#41;]&#40;https://arxiv.org/pdf/1707.09835.pdf&#41;)

[comment]: <> ( - [Bilevel Programming for Hyperparameter Optimization and Meta-Learning&#40;RHG&#41;]&#40;http://export.arxiv.org/pdf/1806.04910&#41;)

[comment]: <> ( - [Truncated Back-propagation for Bilevel Optimization&#40;TG&#41;]&#40;https://arxiv.org/pdf/1810.10667.pdf&#41;)

[comment]: <> ( - [Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace&#40;MT-net&#41;]&#40;http://proceedings.mlr.press/v80/lee18a/lee18a.pdf&#41;)

[comment]: <> ( - [Meta-Learning with warped gradient Descent&#40;WarpGrad&#41;&#41;]&#40;https://arxiv.org/abs/1909.00025&#41;)

[comment]: <> ( - [DARTS: Differentiable Architecture Search&#40;DARTS&#41;]&#40;https://arxiv.org/pdf/1806.09055.pdf&#41;)

[comment]: <> ( - [A Generic First-Order Algorithmic Framework for Bi-Level Programming Beyond Lower-Level Singleton&#40;BDA&#41;]&#40;https://arxiv.org/pdf/2006.04045.pdf&#41;)



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



