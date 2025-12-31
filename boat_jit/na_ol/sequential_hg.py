from typing import List, Dict
from boat_jit.utils import HyperGradientRules, ResultStore
from boat_jit.operation_registry import get_registered_operation


class SequentialHG:
    """
    A class for managing sequential hyper-gradient operations.

    This class dynamically organizes and executes a sequence of hyper-gradient computations
    using user-defined and validated orders of gradient operators.

    Parameters
    ----------
    ordered_instances : List[object]
        A list of instantiated gradient operator objects, ordered as per the adjusted sequence.

    custom_order : List[str]
        The user-defined order of gradient operators.
    """

    def __init__(self, ordered_instances: List[object], custom_order: List[str]):
        self.gradient_instances = ordered_instances
        self.custom_order = custom_order
        self.result_store = ResultStore()  # Use a dedicated result store

    def compute_gradients(self, **kwargs) -> List[Dict]:
        """
        Compute hyper-gradients sequentially using the ordered instances.

        This method processes the hyper-gradients in the defined order, passing intermediate results between consecutive gradient operators.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments required for gradient computations.

        Returns
        -------
        List[Dict]
            A list of dictionaries containing results for each gradient operator.
        """
        self.result_store.clear()  # Reset the result store
        intermediate_result = None

        for idx, gradient_instance in enumerate(self.gradient_instances):
            # Compute the gradient, passing the intermediate result as input
            intermediate_result = gradient_instance.compute_gradients(
                **(kwargs if idx == 0 else intermediate_result),
                next_operation=(
                    self.custom_order[idx + 1]
                    if idx + 1 < len(self.custom_order)
                    else None
                ),
            )

        self.result_store.add(f"gradient_operator_results", intermediate_result)
        return self.result_store.get_results()



def makes_functional_na_operation(custom_order: List[str], **kwargs) -> SequentialHG:
    """
    Dynamically create a SequentialHG object with ordered gradient operators.

    This function validates the user-defined operator order, adjusts it to conform
    with predefined gradient rules, and dynamically loads the corresponding operator classes.

    Parameters
    ----------
    custom_order : List[str]
        The user-defined order of gradient operators.

    **kwargs : dict
        Additional arguments required for initializing gradient operator instances.

    Returns
    -------
    SequentialHG
        An instance of SequentialHG containing the ordered gradient operators and result management.
    """
    # Load the predefined gradient order
    gradient_order = HyperGradientRules.get_gradient_order()

    # Adjust custom order based on predefined gradient order
    adjusted_order = validate_and_adjust_order(custom_order, gradient_order)

    # Dynamically load classes
    gradient_classes = {}
    # module = importlib.import_module("boat.na_ol")
    for op in custom_order:
        gradient_classes[op] = get_registered_operation(op)

    # Reorder classes according to adjusted order
    ordered_instances = [gradient_classes[op](**kwargs) for op in adjusted_order]

    # Return the enhanced sequential hyper-gradient class
    return SequentialHG(ordered_instances, custom_order)


def validate_and_adjust_order(
    custom_order: List[str], gradient_order: List[List[str]]
) -> List[str]:
    """
    Validate and adjust the custom order to match the predefined gradient order.

    This function ensures that the user-defined order adheres to the predefined grouping
    rules and adjusts it accordingly.

    Parameters
    ----------
    custom_order : List[str]
        The user-provided order of gradient operators.

    gradient_order : List[List[str]]
        The predefined order of gradient operator groups.

    Returns
    -------
    List[str]
        Adjusted order of gradient operators following the predefined rules.
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
