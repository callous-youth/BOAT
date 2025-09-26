import pytest
import subprocess
from unittest.mock import patch

dynamic_methodlist = (
    ["NGD"],
    ["DI", "NGD"],
    ["GDA", "NGD"],
    ["GDA", "NGD", "DI"],
    ["DI", "NGD", "GDA"],
)
hyper_methodlist = (
    ["CG"],
    ["CG", "PTT"],
    ["RAD"],
    ["RAD", "PTT"],
    ["RAD", "RGT"],
    ["PTT", "RAD", "RGT"],
    ["FD"],
    ["FD", "PTT"],
    ["NS"],
    ["NS", "PTT"],
    ["IGA"],
    ["IGA", "PTT"],
)
dynamic_method_dm = (["DM"], ["DM", "GDA"])
hyper_method_dm = (["RAD"], ["CG"])
fogm_method = (["VSM"], ["VFM"], ["MESM"], ["PGDM"])


@pytest.mark.parametrize(
    "dynamic_method, hyper_method",
    [
        (dynamic_method, hyper_method)
        for dynamic_method in dynamic_methodlist
        for hyper_method in hyper_methodlist
    ],
)
def test_combination_dynamic_hyper_method(dynamic_method, hyper_method):
    command = [
        "python",
        "/home/runner/work/BOAT/BOAT/examples/data_hyper_cleaning/data_hyper_cleaning.py",
        "--dynamic_method",
        ",".join(dynamic_method),
        "--hyper_method",
        ",".join(hyper_method),
    ]
    print(
        f"Running test with dynamic_method={dynamic_method} and hyper_method={hyper_method}"
    )

    result = subprocess.run(command, capture_output=True, text=True)

    assert (
        result.returncode == 0
    ), f"Test failed for dynamic_method={dynamic_method} and hyper_method={hyper_method}. Error: {result.stderr}"


@pytest.mark.parametrize(
    "dynamic_method, hyper_method",
    [
        (dynamic_method, hyper_method)
        for dynamic_method in dynamic_method_dm
        for hyper_method in hyper_method_dm
    ],
)
def test_combination_dynamic_hyper_method_dm(dynamic_method, hyper_method):
    command = [
        "python",
        "/home/runner/work/BOAT/BOAT/examples/data_hyper_cleaning/data_hyper_cleaning.py",
        "--dynamic_method",
        ",".join(dynamic_method),
        "--hyper_method",
        ",".join(hyper_method),
    ]
    print(
        f"Running test with dynamic_method={dynamic_method} and hyper_method={hyper_method}"
    )

    result = subprocess.run(command, capture_output=True, text=True)

    assert (
        result.returncode == 0
    ), f"Test failed for dynamic_method={dynamic_method} and hyper_method={hyper_method}. Error: {result.stderr}"


@pytest.mark.parametrize("fogm_method", fogm_method)
def test_fogm_method(fogm_method):
    command = [
        "python",
        "/home/runner/work/BOAT/BOAT/examples/data_hyper_cleaning/data_hyper_cleaning.py",
        "--fo_gm",
        fogm_method[0],
    ]
    print(f"Running test with fo_gm={fogm_method}")

    result = subprocess.run(command, capture_output=True, text=True)

    assert (
        result.returncode == 0
    ), f"Test failed for fo_gm={fogm_method}. Error: {result.stderr}"
