from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import os


def load_kernel_from_cu(filepath):
    assert os.path.exists(
        filepath), "Path to CUDA file does not exist: {}".format(filepath)

    cubin_file = os.path.join(os.path.dirname(
        filepath), os.path.basename(filepath).split(".")[0] + ".cubin")

    if os.path.exists(cubin_file):
        # print("Using precompiled version: ", cubin_file)
        return cuda.module_from_file(cubin_file)

    kernel_code_str = ""
    with open(filepath) as f:
        kernel_code_str = "\n".join(f.readlines())
    return SourceModule(kernel_code_str)
