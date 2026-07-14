#include "rms_norm_nvidia.hpp"

#include "../../cuda_common.cuh"

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void rmsNormKernel(T *out, const T *input, const T *weight,
                              size_t rows, size_t width, float epsilon) {
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    extern __shared__ float scratch[];
    float local_sum = 0.0F;
    for (size_t column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = toFloat(input[row * width + column]);
        local_sum += value * value;
    }
    scratch[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(scratch[0] / static_cast<float>(width) + epsilon);
    for (size_t column = threadIdx.x; column < width; column += blockDim.x) {
        out[row * width + column] = fromFloat<T>(
            toFloat(input[row * width + column]) * scale * toFloat(weight[column]));
    }
}

template <typename T>
void launch(std::byte *out, const std::byte *input, const std::byte *weight,
            size_t rows, size_t width, float epsilon) {
    constexpr int threads = 256;
    rmsNormKernel<<<static_cast<int>(rows), threads, threads * sizeof(float)>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(input),
        reinterpret_cast<const T *>(weight), rows, width, epsilon);
    checkKernel("rms norm kernel");
}
} // namespace

void rmsNorm(std::byte *out, const std::byte *input, const std::byte *weight,
             llaisysDataType_t dtype, size_t rows, size_t width, float epsilon) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, input, weight, rows, width, epsilon);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, input, weight, rows, width, epsilon);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, input, weight, rows, width, epsilon);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
