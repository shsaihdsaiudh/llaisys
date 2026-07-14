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
- Transformers elapsed time: 2.09 seconds.
- LLAISYS elapsed time: 0.41 seconds.
- The previously observed 7,494 MiB process peak included the Transformers run
  in the same process and is not reported as an LLAISYS-only peak.

The KV cache is reserved from `input_tokens + max_new_tokens` instead of the
model's 131,072-token maximum context, preventing a fixed 3.5 GiB cache
allocation for short generations.

## NVIDIA optimization follow-up

- Qwen2 inference reuses a model-owned workspace instead of allocating hundreds
  of temporary GPU tensors for every generated token.
- K/V projections write directly into their cache slices, removing two
  synchronous device-to-device copies per layer and inference step.
- Single-token decoding uses fused attention without a global score buffer or
  per-layer `cudaMalloc`/`cudaFree`.
- CUDA argmax uses a 256-thread block reduction instead of scanning the
  151,936-element vocabulary with one thread.
- Cache and workspace storage resize to the current request and release the old
  allocation before growing or shrinking, avoiding retained high-water memory
  and overlapping old/new allocations.

On the same RTX 4090, model, prompt, and 128-token exact-match test, observed
LLAISYS generation time decreased from 0.77 seconds to 0.41 seconds (46.8%).
The final token sequence still matched Transformers exactly.

Additional regression coverage includes the model's real decode geometry
(`12` query heads, `2` KV heads, head dimension `128`) at KV lengths `1`, `256`,
and `257`, a two-layer Qwen2 reference model, cache shrink/grow reuse, duplicate
maxima, NaNs, and the real `151,936`-element vocabulary.

## CPU regression

The tensor test, all operator tests, and the Qwen2 loader/reference test passed
on CPU after the CUDA integration. A separate build with `--nv-gpu=n` also
completed successfully.
