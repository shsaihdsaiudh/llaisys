# LLAISYS Assignment Reproduction Report

## Platform status

| Platform | Runtime | Operators | Qwen2 inference | Status |
| --- | --- | --- | --- | --- |
| CPU (x86_64) | Passed | Passed | Passed | Supported |
| NVIDIA RTX 4090 | Passed | Passed | Passed | Supported |
| Second CUDA-like platform | Pending resource access | Pending | Pending | Not yet verified |

## NVIDIA environment

- GPU: NVIDIA GeForce RTX 4090, 24 GB, compute capability 8.9
- Driver: 570.124.06
- OS: Ubuntu 24.04.1 LTS
- CUDA Toolkit: 12.8
- Xmake: 3.0.9
- Python: 3.12.3
- PyTorch: 2.6.0 NGC build

## Build

```bash
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
xmake f --nv-gpu=y -m release -c
xmake
xmake install -y
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e ./python
```

When building inside a root-owned container, set `XMAKE_ROOT=y` for the Xmake
commands above.

## Runtime and operator verification

```bash
python test/test_runtime.py --device nvidia

for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
    python "test/ops/$op.py" --device nvidia
done

python test/test_qwen2_loader.py --device nvidia
```

All runtime and operator cases passed for Float32, Float16, and BFloat16. The
Qwen2 loader/reference test passed on NVIDIA, including incremental KV-cache
generation.

## End-to-end inference

```bash
python test/test_infer.py \
    --model /data/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --device nvidia \
    --test \
    --max_steps 128 \
    --prompt "Who are you?"
```

Result:

- Generated token sequence matched Transformers exactly.
- Transformers elapsed time: 2.04 seconds.
- LLAISYS elapsed time: 0.77 seconds.
- Observed process peak GPU memory: 7,494 MiB.

The KV cache is reserved from `input_tokens + max_new_tokens` instead of the
model's 131,072-token maximum context, preventing a fixed 3.5 GiB cache
allocation for short generations.

## CPU regression

The tensor test, all operator tests, and the Qwen2 loader/reference test passed
on CPU after the CUDA integration. A separate build with `--nv-gpu=n` also
completed successfully.
