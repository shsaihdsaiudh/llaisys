#pragma once

#include "rope_cuda.hpp"

#include "../../cuda/common.cuh"

namespace llaisys::ops::cuda {
namespace {
template <typename T>
__global__ void ropeKernel(T *out, const T *input, const int64_t *positions,
                           size_t sequence_length, size_t heads,
                           size_t head_dimension, float theta) {
    const size_t half = head_dimension / 2;
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = sequence_length * heads * half;
    if (index >= count) {
        return;
    }
    const size_t pair_index = index % half;
    const size_t token_head = index / half;
    const size_t token = token_head / heads;
    const size_t base = token_head * head_dimension;
    const float divisor = powf(theta, 2.0F * static_cast<float>(pair_index)
                                          / static_cast<float>(head_dimension));
    const float angle = static_cast<float>(positions[token]) / divisor;
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    const float first = toFloat(input[base + pair_index]);
    const float second = toFloat(input[base + half + pair_index]);
    out[base + pair_index] = fromFloat<T>(first * cosine - second * sine);
    out[base + half + pair_index] = fromFloat<T>(second * cosine + first * sine);
}

template <typename T>
void launch(std::byte *out, const std::byte *input, const int64_t *positions,
            size_t sequence_length, size_t heads, size_t head_dimension, float theta) {
    constexpr int threads = 256;
    const size_t count = sequence_length * heads * (head_dimension / 2);
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    ropeKernel<<<blocks, threads>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(input),
                                    positions, sequence_length, heads, head_dimension, theta);
    checkKernel("rope kernel");
}
} // namespace

void rope(std::byte *out, const std::byte *input, const int64_t *positions,
          llaisysDataType_t dtype, size_t sequence_length, size_t heads,
          size_t head_dimension, float theta) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, input, positions, sequence_length, heads, head_dimension, theta);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, input, positions, sequence_length, heads, head_dimension, theta);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, input, positions, sequence_length, heads, head_dimension, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cuda
