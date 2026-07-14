#include "argmax_nvidia.hpp"

#include "../../cuda_common.cuh"

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void argmaxKernel(int64_t *max_index, T *max_value, const T *values, size_t count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    size_t best_index = 0;
    float best_value = toFloat(values[0]);
    for (size_t i = 1; i < count; ++i) {
        const float value = toFloat(values[i]);
        if (value > best_value) {
            best_index = i;
            best_value = value;
        }
    }
    max_index[0] = static_cast<int64_t>(best_index);
    max_value[0] = values[best_index];
}

template <typename T>
void launch(int64_t *max_index, std::byte *max_value, const std::byte *values, size_t count) {
    argmaxKernel<<<1, 1>>>(max_index, reinterpret_cast<T *>(max_value),
                           reinterpret_cast<const T *>(values), count);
    checkKernel("argmax kernel");
}
} // namespace

void argmax(int64_t *max_index, std::byte *max_value, const std::byte *values,
            llaisysDataType_t dtype, size_t count) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(max_index, max_value, values, count);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(max_index, max_value, values, count);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(max_index, max_value, values, count);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
