import subprocess
import sys
from pathlib import Path

import pytest

DATA_HYPER_CLEANING_SCRIPT = Path(__file__).with_name("data_hyper_cleaning.py")


def run_data_hyper_cleaning(*args):
    command = [sys.executable, str(DATA_HYPER_CLEANING_SCRIPT), *args]
    return subprocess.run(command, capture_output=True, text=True)

gm_oplist = (
    ["NGD"],
    ["DI", "NGD"],
    ["GDA", "NGD"],
    ["GDA", "NGD", "DI"],
    ["DI", "NGD", "GDA"],
)
na_oplist = (
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
gm_op_dm = (["DM","NGD"], ["DM","GDA","NGD"])
na_op_dm = (["RAD"], ["CG"])
fo_ol_method = (["VSO"], ["VFO"], ["MESO"], ["PGDO"])


@pytest.mark.parametrize(
    "gm_op, na_op",
    [
        (gm_op, na_op)
        for gm_op in gm_oplist
        for na_op in na_oplist
    ],
)
def test_combination_dynamic_na_op(gm_op, na_op):
    print(
        f"Running test with gm_op={gm_op} and na_op={na_op}"
    )

    result = run_data_hyper_cleaning(
        "--gm_op",
        ",".join(gm_op),
        "--na_op",
        ",".join(na_op),
    )

    assert (
        result.returncode == 0
    ), f"Test failed for gm_op={gm_op} and na_op={na_op}. Error: {result.stderr}"


@pytest.mark.parametrize(
    "gm_op, na_op",
    [
        (gm_op, na_op)
        for gm_op in gm_op_dm
        for na_op in na_op_dm
    ],
)
def test_combination_dynamic_na_op_dm(gm_op, na_op):
    print(
        f"Running test with gm_op={gm_op} and na_op={na_op}"
    )

    result = run_data_hyper_cleaning(
        "--gm_op",
        ",".join(gm_op),
        "--na_op",
        ",".join(na_op),
    )

    assert (
        result.returncode == 0
    ), f"Test failed for gm_op={gm_op} and na_op={na_op}. Error: {result.stderr}"


@pytest.mark.parametrize("fo_ol_method", fo_ol_method)
def test_fo_ol_method(fo_ol_method):
    print(f"Running test with fo_op={fo_ol_method}")

    result = run_data_hyper_cleaning("--fo_op", fo_ol_method[0])

    assert (
        result.returncode == 0
    ), f"Test failed for fo_op={fo_ol_method}. Error: {result.stderr}"
