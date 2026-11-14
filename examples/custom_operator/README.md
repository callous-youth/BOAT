# Add a custom hypergradient operator

This example extends `RAD` with gradient clipping. From the repository root,
after installing BOAT, run the following command.

```bash
python -m examples.custom_operator.clipped_rad
```

The example runs on CPU and uses scalar objectives in place of a dataset.

## 1. Define and register the operator

```python
@register_class
class CLIPPED_RAD(RAD):
    def compute_gradients(self, *args, **kwargs):
        result = super().compute_gradients(*args, **kwargs)
        clip_grad_norm_(self.ul_var, max_norm=0.5)
        return result
```

`RAD` writes the upper hypergradient to the upper parameters' `.grad` fields.
`CLIPPED_RAD` limits their total norm to `0.5` and passes RAD's result back to
BOAT. When the module loads, `@register_class` registers the class by name.

## 2. Select it in BOAT

Add `CLIPPED_RAD` to `HyperGradientRules` before building the solver, then use
`gm_op=["NGD"]` and `na_op=["CLIPPED_RAD"]` in the `Problem` config.

```python
order = HyperGradientRules.get_gradient_order()
HyperGradientRules.set_gradient_order(order + [["CLIPPED_RAD"]])
```

Call `problem.set_track_trajectory(True)` before running an iteration so the
custom operator can differentiate through the lower update. BOAT enables
tracking automatically for the built-in name `RAD`. The script restores the
original rule order after the demo.

## 3. Check the result

The lower loss is `(y - x)^2 / 2`. One lower SGD step with learning rate `1`
gives `y = x`, so the upper loss `(y - 2)^2 / 2` has hypergradient `x - 2`.
Each run starts from `x = y = 0` and uses `return_grad=True` to read the
gradient before the upper optimizer steps.

```text
RAD:         -2.000000
CLIPPED_RAD: -0.500000
```

The complete [example script](clipped_rad.py) contains the models, losses,
and solver configuration used for both runs.
