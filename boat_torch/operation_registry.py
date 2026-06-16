Operation_REGISTRY = {}


def register_class(cls):
    """
    Register a new operation class to the global registry.
    """
    Operation_REGISTRY[cls.__name__] = cls
    return cls


def get_registered_operation(name):
    """
    Retrieve a registered operation class by name.
    """
    if name not in Operation_REGISTRY:
        raise ValueError(f"Class '{name}' is not registered.")
    return Operation_REGISTRY[name]
