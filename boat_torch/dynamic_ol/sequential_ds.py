from typing import List, Dict
from boat_torch.utils import DynamicalSystemRules, ResultStore
from boat_torch.operation_registry import get_registered_operation

importlib = __import__("importlib")


class SequentialDS:
    """
    A dynamically created class for sequential hyper-gradient operations.

    Attributes
    ----------
    gradient_instances : List[object]
        A list of gradient operator instances, each implementing an `optimize` method.
    custom_order : List[str]
        A custom-defined order for executing the gradient operators.
    result_store : ResultStore
        An instance of the `ResultStore` class for storing intermediate and final results.
    """

    def __init__(self, ordered_instances: List[object], custom_order: List[str]):
        """
        Initialize the SequentialDS class with gradient operator instances and a custom execution order.

        Parameters
        ----------
        ordered_instances : List[object]
            A list of gradient operator instances to be executed sequentially.
        custom_order : List[str]
            A list defining the custom execution order of the gradient operators.
        """
        self.gradient_instances = ordered_instances
        self.custom_order = custom_order
        self.result_store = ResultStore()  # Use a dedicated result store

    def optimize(self, **kwargs) -> List[Dict]:
        """
        Compute gradients sequentially using the ordered gradient operator instances.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments required for gradient computations.

        Returns
        -------
        List[Dict]
            A list of dictionaries containing results for each gradient operator.

        Notes
        -----
        - The results of each gradient operator are passed as inputs to the subsequent operator.
        - Results are stored in the `ResultStore` instance for further use or analysis.

        Example
        -------
        >>> gradient_instances = [GradientOp1(), GradientOp2()]
        >>> custom_order = ["op1", "op2"]
        >>> sequential_ds = SequentialDS(gradient_instances, custom_order)
        >>> results = sequential_ds.optimize(input_data=data)
        """
        self.result_store.clear()  # Reset the result store
        intermediate_result = None

        for idx, gradient_instance in enumerate(self.gradient_instances):
            # Compute the gradient, passing the intermediate result as input
            intermediate_result = gradient_instance.optimize(
                **(kwargs if idx == 0 else intermediate_result),
                next_operation=(
                    self.custom_order[idx + 1]
                    if idx + 1 < len(self.custom_order)
                    else None
                ),
            )
            # Store the result
            # self.result_store.add(f"dynamic_results_{idx}", result)
            # intermediate_result = result
        self.result_store.add(f"dynamic_results_{idx}", intermediate_result)
        return self.result_store.get_results()


def makes_functional_dynamical_system(
    custom_order: List[str], **kwargs
) -> SequentialDS:
    """
    Dynamically create a SequentialHyperGradient object with ordered gradient operators.

    Parameters
    ----------
    custom_order : List[str]
        User-defined operator order.

    Returns
    -------
    SequentialHyperGradient
        An instance with ordered gradient operators and result management.
    """
    # Load the predefined gradient order
    gradient_order = DynamicalSystemRules.get_gradient_order()

    # Adjust custom order based on predefined gradient order
    adjusted_order = validate_and_adjust_order(custom_order, gradient_order)

    # Dynamically load classes
    gradient_classes = {}
    # module = importlib.import_module("boat.dynamic_ol")
    for op in custom_order:
        gradient_classes[op] = get_registered_operation(op)

    # Reorder classes according to adjusted order
    ordered_instances = [gradient_classes[op](**kwargs) for op in adjusted_order]

    # Return the enhanced sequential hyper-gradient class
    return SequentialDS(ordered_instances, custom_order)


def validate_and_adjust_order(
    custom_order: List[str], gradient_order: List[List[str]]
) -> List[str]:
    """
    Validate and adjust the custom order to align with the predefined gradient operator groups.

    Parameters
    ----------
    custom_order : List[str]
        The user-defined order of gradient operators.
    gradient_order : List[List[str]]
        The predefined grouping of gradient operators, specifying valid order constraints.

    Returns
    -------
    List[str]
        A validated and adjusted list of gradient operators that conforms to the predefined order.

    Notes
    -----
    - The function filters out invalid operators from `custom_order` that do not exist in `gradient_order`.
    - It ensures that the returned order follows the precedence rules defined in `gradient_order`.

    Example
    -------
    >>> custom_order = ["op1", "op3", "op2"]
    >>> gradient_order = [["op1", "op2"], ["op3"]]
    >>> adjusted_order = validate_and_adjust_order(custom_order, gradient_order)
    >>> print(adjusted_order)
    ['op1', 'op2', 'op3']
    """
    # Create a set of valid operators for quick lookup
    valid_operators = {op for group in gradient_order for op in group}

    # Filter out invalid operators
    custom_order = [op for op in custom_order if op in valid_operators]

    # Adjust order to follow gradient_order
    adjusted_order = []
    for group in gradient_order:
        for op in group:
            if op in custom_order:
                adjusted_order.append(op)

    return adjusted_order
