import os

import pytest
import subprocess
from unittest.mock import patch

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
fo_ol_method = (["VSO"], ["VFO"], ["MESO"], ["PGDO"], ["ALTO"])


@pytest.mark.parametrize(
    "gm_op, na_op",
    [
        (gm_op, na_op)
        for gm_op in gm_oplist
        for na_op in na_oplist
    ],
)
def test_combination_dynamic_na_op(gm_op, na_op):
    command = [
        "python",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data_hyper_cleaning.py"
        ),
        "--gm_op",
        ",".join(gm_op),
        "--na_op",
        ",".join(na_op),
    ]
    print(
        f"Running test with gm_op={gm_op} and na_op={na_op}"
    )

    result = subprocess.run(command, capture_output=True, text=True)

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
    command = [
        "python",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data_hyper_cleaning.py"
        ),
        "--gm_op",
        ",".join(gm_op),
        "--na_op",
        ",".join(na_op),
    ]
    print(
        f"Running test with gm_op={gm_op} and na_op={na_op}"
    )

    result = subprocess.run(command, capture_output=True, text=True)

    assert (
        result.returncode == 0
    ), f"Test failed for gm_op={gm_op} and na_op={na_op}. Error: {result.stderr}"


@pytest.mark.parametrize("fo_ol_method", fo_ol_method)
def test_fo_ol_method(fo_ol_method):
    command = [
        "python",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data_hyper_cleaning.py"
        ),
        "--fo_op",
        fo_ol_method[0],
    ]
    print(f"Running test with fo_op={fo_ol_method}")

    result = subprocess.run(command, capture_output=True, text=True)

    assert (
        result.returncode == 0
    ), f"Test failed for fo_op={fo_ol_method}. Error: {result.stderr}"


def test_subprocess_uses_portable_script_path(monkeypatch):
    """Regression: every subprocess call must reference a script path that
    resolves relative to this test file, not a hard-coded CI absolute path."""
    captured = []

    def fake_run(command, *args, **kwargs):
        captured.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    expected = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data_hyper_cleaning.py")
    assert os.path.isfile(expected)

    test_combination_dynamic_na_op(["NGD"], ["CG"])
    test_combination_dynamic_na_op_dm(["DM", "NGD"], ["CG"])
    test_fo_ol_method("VSO")

    assert len(captured) == 3
    for command in captured:
        script = command[1]
        assert os.path.isabs(script), (
            f"Expected an absolute portable path, got {script!r}"
        )
        assert os.path.normpath(script) == os.path.normpath(expected)
