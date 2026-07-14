#include "add_nvidia.hpp"

#include "../../cuda_common.cuh"

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void addKernel(T *out, const T *a, const T *b, size_t count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        out[index] = fromFloat<T>(toFloat(a[index]) + toFloat(b[index]));
    }
}

template <typename T>
void launch(std::byte *out, const std::byte *a, const std::byte *b, size_t count) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    addKernel<<<blocks, threads>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(a),
                                  reinterpret_cast<const T *>(b), count);
    checkKernel("add kernel");
}
} // namespace

void add(std::byte *out, const std::byte *a, const std::byte *b, llaisysDataType_t dtype, size_t count) {
    if (count == 0) {
        return;
    }
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, a, b, count);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, a, b, count);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, a, b, count);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
