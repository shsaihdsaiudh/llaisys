#include "embedding_nvidia.hpp"

#include "../../cuda_common.cuh"

namespace llaisys::ops::cuda {
namespace {
template <typename T>
__global__ void embeddingKernel(T *out, const int64_t *indices, const T *weight,
                                size_t rows, size_t width) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = rows * width;
    if (index < count) {
        const size_t row = index / width;
        const size_t column = index % width;
        out[index] = weight[static_cast<size_t>(indices[row]) * width + column];
    }
}

template <typename T>
void launch(std::byte *out, const int64_t *indices, const std::byte *weight,
            size_t rows, size_t width) {
    constexpr int threads = 256;
    const size_t count = rows * width;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    embeddingKernel<<<blocks, threads>>>(reinterpret_cast<T *>(out), indices,
                                         reinterpret_cast<const T *>(weight), rows, width);
    checkKernel("embedding kernel");
}
} // namespace

void embedding(std::byte *out, const int64_t *indices, const std::byte *weight,
               llaisysDataType_t dtype, size_t rows, size_t width) {
    if (rows == 0 || width == 0) {
        return;
    }
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, indices, weight, rows, width);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, indices, weight, rows, width);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, indices, weight, rows, width);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cuda
