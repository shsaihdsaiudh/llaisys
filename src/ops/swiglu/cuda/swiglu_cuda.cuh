#pragma once

#include "swiglu_cuda.hpp"

#include "../../cuda/common.cuh"

namespace llaisys::ops::cuda {
namespace {
template <typename T>
__global__ void swigluKernel(T *out, const T *gate, const T *up, size_t count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float gate_value = toFloat(gate[index]);
        const float up_value = toFloat(up[index]);
        const float silu = gate_value / (1.0F + expf(-gate_value));
        out[index] = fromFloat<T>(up_value * silu);
    }
}

template <typename T>
void launch(std::byte *out, const std::byte *gate, const std::byte *up, size_t count) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    swigluKernel<<<blocks, threads>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(gate),
                                     reinterpret_cast<const T *>(up), count);
    checkKernel("swiglu kernel");
}
} // namespace

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t count) {
    if (count == 0) {
        return;
    }
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, gate, up, count);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, gate, up, count);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, gate, up, count);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cuda
