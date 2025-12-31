import os
import time
import shutil
import subprocess
import pytest
import platform


gm_oplist = (["NGD"], ["NGD", "GDA"])
na_oplist = (
    ["IAD"],
    ["IAD", "PTT"],
    ["CG", "IAD"],
    ["CG", "IAD", "PTT"],
    ["NS", "IAD"],
    ["NS", "IAD", "PTT"],
    ["FOA", "IAD"],
    ["FOA", "IAD", "PTT"],
)
fo_ol_method = (["VSO"], ["VFO"], ["MESO"], ["PGDO"])


t0 = time.strftime("%Y_%m_%d_%H_%M_%S")
args = "meta_learning/method_test"


base_folder = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(base_folder, args, t0)


if not os.path.exists(folder):
    os.makedirs(folder)


ganfolder = os.path.join(folder, "meta_learning.py")
shutil.copyfile(os.path.join(base_folder, "meta_learning.py"), ganfolder)


script_extension = ".bat" if platform.system() == "Windows" else ".sh"
script_file = os.path.join(folder, "set" + script_extension)


with open(script_file, "w") as f:
    k = 0
    for gm_op in gm_oplist:
        for na_op in na_oplist:
            k += 1
            f.write(
                f'python /home/runner/work/BOAT/BOAT/examples/meta_learning/meta_learning.py --gm_op {",".join(gm_op)} --na_op {",".join(na_op)} \n'
            )


if platform.system() != "Windows":
    os.chmod(script_file, 0o775)



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
        "/home/runner/work/BOAT/BOAT/examples/meta_learning/meta_learning.py",
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
        "/home/runner/work/BOAT/BOAT/examples/meta_learning/meta_learning.py",
        "--fo_op",
        fo_ol_method[0],
    ]
    print(f"Running test with fo_op={fo_ol_method}")

    result = subprocess.run(command, capture_output=True, text=True)

    assert (
        result.returncode == 0
    ), f"Test failed for fo_op={fo_ol_method}. Error: {result.stderr}"
