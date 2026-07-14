#pragma once

#include "self_attention_cuda.hpp"

#include "../../cuda/common.cuh"

#include <cmath>

namespace llaisys::ops::cuda {
namespace {
template <typename T>
__device__ float attentionScore(const T *query, const T *key,
                                size_t query_base, size_t key_base,
                                size_t head_dimension, float scale) {
    float result = 0.0F;
    for (size_t column = 0; column < head_dimension; ++column) {
        result += toFloat(query[query_base + column]) * toFloat(key[key_base + column]);
    }
    return result * scale;
}

template <typename T>
__global__ void attentionScoresKernel(float *scores, const T *query, const T *key,
                                      size_t query_length, size_t kv_length,
                                      size_t query_heads, size_t kv_heads,
                                      size_t head_dimension, float scale) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = query_length * query_heads * kv_length;
    if (index >= count) {
        return;
    }
    const size_t key_position = index % kv_length;
    const size_t query_head_index = index / kv_length;
    const size_t query_head = query_head_index % query_heads;
    const size_t query_position = query_head_index / query_heads;
    const size_t cached_tokens = kv_length - query_length;
    if (key_position >= cached_tokens + query_position + 1) {
        scores[index] = -INFINITY;
        return;
    }
    const size_t kv_head = query_head / (query_heads / kv_heads);
    const size_t query_base = (query_position * query_heads + query_head) * head_dimension;
    const size_t key_base = (key_position * kv_heads + kv_head) * head_dimension;
    scores[index] = attentionScore(query, key, query_base, key_base, head_dimension, scale);
}

__global__ void softmaxKernel(float *scores, size_t rows, size_t columns) {
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    extern __shared__ float scratch[];
    float local_max = -INFINITY;
    for (size_t column = threadIdx.x; column < columns; column += blockDim.x) {
        local_max = fmaxf(local_max, scores[row * columns + column]);
    }
    scratch[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float maximum = scratch[0];
    float local_sum = 0.0F;
    for (size_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const float probability = expf(scores[row * columns + column] - maximum);
        scores[row * columns + column] = probability;
        local_sum += probability;
    }
    scratch[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denominator = scratch[0];
    for (size_t column = threadIdx.x; column < columns; column += blockDim.x) {
        scores[row * columns + column] /= denominator;
    }
}

template <typename T>
__global__ void attentionValuesKernel(T *out, const float *scores, const T *value,
                                      size_t query_length, size_t kv_length,
                                      size_t query_heads, size_t kv_heads,
                                      size_t value_dimension) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = query_length * query_heads * value_dimension;
    if (index >= count) {
        return;
    }
    const size_t column = index % value_dimension;
    const size_t query_head_index = index / value_dimension;
    const size_t query_head = query_head_index % query_heads;
    const size_t kv_head = query_head / (query_heads / kv_heads);
    float result = 0.0F;
    for (size_t key_position = 0; key_position < kv_length; ++key_position) {
        const float probability = scores[query_head_index * kv_length + key_position];
        const size_t value_index = (key_position * kv_heads + kv_head) * value_dimension + column;
        result += probability * toFloat(value[value_index]);
    }
    out[index] = fromFloat<T>(result);
}

template <typename T>
__global__ void decodeAttentionKernel(T *out, const T *query, const T *key,
                                      const T *value, size_t kv_length,
                                      size_t query_heads, size_t kv_heads,
                                      size_t head_dimension, size_t value_dimension,
                                      float scale) {
    const size_t query_head = blockIdx.x;
    if (query_head >= query_heads) {
        return;
    }

    extern __shared__ float scratch[];
    __shared__ float maximum;
    __shared__ float denominator;

    const size_t kv_head = query_head / (query_heads / kv_heads);
    const size_t query_base = query_head * head_dimension;

    float local_max = -INFINITY;
    for (size_t key_position = threadIdx.x; key_position < kv_length;
         key_position += blockDim.x) {
        const size_t key_base = (key_position * kv_heads + kv_head) * head_dimension;
        local_max = fmaxf(
            local_max,
            attentionScore(query, key, query_base, key_base, head_dimension, scale));
    }
    scratch[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        maximum = scratch[0];
    }
    __syncthreads();

    float local_sum = 0.0F;
    for (size_t key_position = threadIdx.x; key_position < kv_length;
         key_position += blockDim.x) {
        const size_t key_base = (key_position * kv_heads + kv_head) * head_dimension;
        local_sum += expf(
            attentionScore(query, key, query_base, key_base, head_dimension, scale)
            - maximum);
    }
    scratch[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        denominator = scratch[0];
    }
    __syncthreads();

    float result = 0.0F;
    for (size_t chunk_start = 0; chunk_start < kv_length; chunk_start += blockDim.x) {
        const size_t key_position = chunk_start + threadIdx.x;
        if (key_position < kv_length) {
            const size_t key_base = (key_position * kv_heads + kv_head) * head_dimension;
            scratch[threadIdx.x] = expf(
                attentionScore(query, key, query_base, key_base, head_dimension, scale)
                - maximum)
                                 / denominator;
        }
        __syncthreads();

        if (threadIdx.x < value_dimension) {
            const size_t remaining = kv_length - chunk_start;
            const size_t chunk_size = remaining < blockDim.x ? remaining : blockDim.x;
            for (size_t offset = 0; offset < chunk_size; ++offset) {
                const size_t value_index =
                    ((chunk_start + offset) * kv_heads + kv_head) * value_dimension
                    + threadIdx.x;
                result += scratch[offset] * toFloat(value[value_index]);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < value_dimension) {
        out[query_head * value_dimension + threadIdx.x] = fromFloat<T>(result);
    }
}

template <typename T>
void launchDecode(std::byte *out, const std::byte *query, const std::byte *key,
                  const std::byte *value, size_t kv_length, size_t query_heads,
                  size_t kv_heads, size_t head_dimension, size_t value_dimension,
                  float scale) {
    constexpr int threads = 256;
    decodeAttentionKernel<<<static_cast<int>(query_heads), threads, threads * sizeof(float)>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(query),
        reinterpret_cast<const T *>(key), reinterpret_cast<const T *>(value),
        kv_length, query_heads, kv_heads, head_dimension, value_dimension, scale);
    checkKernel("decode attention kernel");
}

template <typename T>
void launch(std::byte *out, const std::byte *query, const std::byte *key,
            const std::byte *value, size_t query_length, size_t kv_length,
            size_t query_heads, size_t kv_heads, size_t head_dimension,
            size_t value_dimension, float scale) {
    constexpr size_t decode_threads = 256;
    if (query_length == 1 && value_dimension <= decode_threads) {
        return launchDecode<T>(out, query, key, value, kv_length, query_heads,
                               kv_heads, head_dimension, value_dimension, scale);
    }

    constexpr int threads = 256;
    const size_t score_count = query_length * query_heads * kv_length;
    float *scores = nullptr;
    checkCuda(cudaMalloc(&scores, score_count * sizeof(float)), "cudaMalloc attention scores");
    try {
        const int score_blocks = static_cast<int>((score_count + threads - 1) / threads);
        attentionScoresKernel<<<score_blocks, threads>>>(
            scores, reinterpret_cast<const T *>(query), reinterpret_cast<const T *>(key),
            query_length, kv_length, query_heads, kv_heads, head_dimension, scale);
        checkKernel("attention scores kernel");

        const size_t score_rows = query_length * query_heads;
        softmaxKernel<<<static_cast<int>(score_rows), threads, threads * sizeof(float)>>>(
            scores, score_rows, kv_length);
        checkKernel("attention softmax kernel");

        const size_t output_count = query_length * query_heads * value_dimension;
        const int output_blocks = static_cast<int>((output_count + threads - 1) / threads);
        attentionValuesKernel<<<output_blocks, threads>>>(
            reinterpret_cast<T *>(out), scores, reinterpret_cast<const T *>(value),
            query_length, kv_length, query_heads, kv_heads, value_dimension);
        checkKernel("attention values kernel");
        checkCuda(cudaFree(scores), "cudaFree attention scores");
    } catch (...) {
        cudaFree(scores);
        throw;
    }
}
} // namespace

void selfAttention(std::byte *out, const std::byte *query, const std::byte *key,
                   const std::byte *value, llaisysDataType_t dtype,
                   size_t query_length, size_t kv_length, size_t query_heads,
                   size_t kv_heads, size_t head_dimension, size_t value_dimension,
                   float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, query, key, value, query_length, kv_length,
                             query_heads, kv_heads, head_dimension, value_dimension, scale);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, query, key, value, query_length, kv_length,
                              query_heads, kv_heads, head_dimension, value_dimension, scale);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, query, key, value, query_length, kv_length,
                                     query_heads, kv_heads, head_dimension, value_dimension, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cuda
