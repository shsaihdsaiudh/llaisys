#pragma once

#include "../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace llaisys::ops::nvidia {
inline void checkCuda(cudaError_t status, const char *operation) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

inline void checkKernel(const char *operation) {
    checkCuda(cudaGetLastError(), operation);
}

template <typename T>
__device__ inline float toFloat(T value) {
    return static_cast<float>(value);
}

template <>
__device__ inline float toFloat<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ inline float toFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ inline T fromFloat(float value) {
    return static_cast<T>(value);
}

template <>
__device__ inline __half fromFloat<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ inline __nv_bfloat16 fromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}
} // namespace llaisys::ops::nvidia
