import llaisys
import torch
from test_utils import *
import argparse
import gc
from concurrent.futures import ThreadPoolExecutor


def test_basic_runtime_api(device_name: str = "cpu"):

    api = llaisys.RuntimeAPI(llaisys_device(device_name))

    ndev = api.get_device_count()
    print(f"Found {ndev} {device_name} devices")
    if ndev == 0:
        print("     Skipped")
        return

    for i in range(ndev):
        print("Testing device {i}...")
        api.set_device(i)
        test_memcpy(api, 1024 * 1024)

        print("     Passed")


def test_memcpy(api, size_bytes: int):
    a = torch.zeros((size_bytes,), dtype=torch.uint8, device=torch_device("cpu"))
    b = torch.ones_like(a)
    device_a = api.malloc_device(size_bytes)
    try:
        device_b = api.malloc_device(size_bytes)
        try:
            # a -> device_a
            api.memcpy_sync(
                device_a,
                a.data_ptr(),
                size_bytes,
                llaisys.MemcpyKind.H2D,
            )
            # device_a -> device_b
            api.memcpy_sync(
                device_b,
                device_a,
                size_bytes,
                llaisys.MemcpyKind.D2D,
            )
            # device_b -> b
            api.memcpy_sync(
                b.data_ptr(),
                device_b,
                size_bytes,
                llaisys.MemcpyKind.D2H,
            )
        finally:
            api.free_device(device_b)
    finally:
        api.free_device(device_a)

    torch.testing.assert_close(a, b)


def test_tensor_outlives_creator_thread(device_name: str):
    device = llaisys_device(device_name)
    expected_host = torch.arange(16, dtype=torch.int64)
    expected = expected_host.to(torch_device(device_name))

    def create_tensor():
        llaisys.set_context_runtime(device)
        tensor = llaisys.Tensor(
            expected.shape, dtype=llaisys.DataType.I64, device=device
        )
        tensor.load(expected_host.data_ptr())
        return tensor

    with ThreadPoolExecutor(max_workers=1) as executor:
        tensor = executor.submit(create_tensor).result()

    assert check_equal(tensor, expected)
    del tensor
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "metax"], type=str)
    args = parser.parse_args()
    test_basic_runtime_api(args.device)
    test_tensor_outlives_creator_thread(args.device)
    
    print("\033[92mTest passed!\033[0m\n")
