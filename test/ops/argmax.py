import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, check_equal, benchmark, zero_tensor


def torch_argmax(max_idx, max_val, vals):
    torch.max(vals, keepdim=True, dim=-1, out=(max_val, max_idx))


def test_op_argmax(
    shape,
    dtype_name="f32",
    device_name="cpu",
    profile=False,
    force_tie=False,
    force_nan=False,
):
    case_suffix = " (tie)" if force_tie else " (NaN)" if force_nan else ""
    print(f"   shape {shape} dtype <{dtype_name}>{case_suffix}")
    vals, vals_ = random_tensor(shape, dtype_name, device_name)
    max_idx, max_idx_ = zero_tensor((1,), "i64", device_name)
    max_val, max_val_ = zero_tensor((1,), dtype_name, device_name)

    if force_tie or force_nan:
        vals.zero_()
        vals[1] = float("nan") if force_nan else 1
        vals[-1] = vals[1]
        api = llaisys.RuntimeAPI(vals_.device_type())
        api.memcpy_sync(
            vals_.data_ptr(),
            vals.data_ptr(),
            vals.numel() * vals.element_size(),
            llaisys.MemcpyKind.D2D,
        )

    torch_argmax(max_idx, max_val, vals)
    llaisys.Ops.argmax(max_idx_, max_val_, vals_)

    if force_nan:
        actual_max_val = torch.empty_like(max_val)
        api.memcpy_sync(
            actual_max_val.data_ptr(),
            max_val_.data_ptr(),
            actual_max_val.element_size(),
            llaisys.MemcpyKind.D2D,
        )
        assert torch.isnan(actual_max_val).all()
        assert check_equal(max_idx_, max_idx, strict=True)
    else:
        assert check_equal(max_val_, max_val, strict=True)
        assert check_equal(max_idx_, max_idx, strict=True)

    if profile:
        benchmark(
            lambda: torch_argmax(max_idx, max_val, vals),
            lambda: llaisys.Ops.argmax(max_idx_, max_val_, vals_),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    test_shapes = [(4,), (4096,)]
    if args.device == "nvidia" or args.profile:
        test_shapes.append((151936,))
    test_dtypes = ["f32", "f16", "bf16"]
    print(f"Testing Ops.argmax on {args.device}")
    for shape in test_shapes:
        for dtype_name in test_dtypes:
            test_op_argmax(shape, dtype_name, args.device, args.profile)

    for dtype_name in test_dtypes:
        test_op_argmax((257,), dtype_name, args.device, force_tie=True)
        if args.device == "nvidia":
            test_op_argmax((257,), dtype_name, args.device, force_nan=True)

    print("\033[92mTest passed!\033[0m\n")
