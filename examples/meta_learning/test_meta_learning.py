import os
import time
import shutil
import subprocess
import pytest
import platform

# 假设 meta_learning.py 已经被复制到正确的位置
dynamic_methodlist = (["NGD"], ["NGD", "GDA"])
hyper_methodlist = (
    ["IAD"],
    ["IAD", "PTT"],
    ["CG", "IAD"],
    ["CG", "IAD", "PTT"],
    ["NS", "IAD"],
    ["NS", "IAD", "PTT"],
    ["FOA", "IAD"],
    ["FOA", "IAD", "PTT"],
)
fogm_method = (["VSM"], ["VFM"], ["MESM"], ["PGDM"])

# 获取当前时间
t0 = time.strftime("%Y_%m_%d_%H_%M_%S")
args = "meta_learning/method_test"  # 使用相对路径

# 获取当前脚本所在的目录（相对路径）
base_folder = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的绝对路径
folder = os.path.join(base_folder, args, t0)  # 构建相对路径

# 创建文件夹
if not os.path.exists(folder):
    os.makedirs(folder)

# 将 Python 文件复制到目标文件夹
ganfolder = os.path.join(folder, "meta_learning.py")
shutil.copyfile(os.path.join(base_folder, "meta_learning.py"), ganfolder)

# 创建一个临时的 shell 脚本（Windows 下是 .bat 文件）
script_extension = ".bat" if platform.system() == "Windows" else ".sh"
script_file = os.path.join(folder, "set" + script_extension)

# 创建批处理或 shell 脚本
with open(script_file, "w") as f:
    k = 0
    for dynamic_method in dynamic_methodlist:
        for hyper_method in hyper_methodlist:
            k += 1
            f.write(
                f'python /home/runner/work/BOAT/BOAT/examples/meta_learning/meta_learning.py --dynamic_method {",".join(dynamic_method)} --hyper_method {",".join(hyper_method)} \n'
            )

# 如果是 Ubuntu 系统, 使得脚本具有执行权限
if platform.system() != "Windows":
    os.chmod(script_file, 0o775)  # 给 sh 文件执行权限


# 使用 pytest.mark.parametrize 进行参数化
@pytest.mark.parametrize(
    "dynamic_method, hyper_method",
    [
        (dynamic_method, hyper_method)
        for dynamic_method in dynamic_methodlist
        for hyper_method in hyper_methodlist
    ],
)
def test_combination_dynamic_hyper_method(dynamic_method, hyper_method):
    # 构建命令
    command = [
        "python",
        "/home/runner/work/BOAT/BOAT/examples/meta_learning/meta_learning.py",
        "--dynamic_method",
        ",".join(dynamic_method),
        "--hyper_method",
        ",".join(hyper_method),
    ]
    print(
        f"Running test with dynamic_method={dynamic_method} and hyper_method={hyper_method}"
    )

    result = subprocess.run(command, capture_output=True, text=True)

    # 确保命令执行成功
    assert (
        result.returncode == 0
    ), f"Test failed for dynamic_method={dynamic_method} and hyper_method={hyper_method}. Error: {result.stderr}"

@pytest.mark.parametrize("fogm_method", fogm_method)
def test_fogm_method(fogm_method):
    command = [
        "python",
        "/home/runner/work/BOAT/BOAT/examples/meta_learning/meta_learning.py",
        "--fo_gm",
        fogm_method[0],
    ]
    print(f"Running test with fo_gm={fogm_method}")

    result = subprocess.run(command, capture_output=True, text=True)

    assert (
        result.returncode == 0
    ), f"Test failed for fo_gm={fogm_method}. Error: {result.stderr}"
