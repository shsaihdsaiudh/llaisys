#include "argmax_nvidia.hpp"

#include "../../cuda_common.cuh"

namespace llaisys::ops::cuda {
namespace {
constexpr unsigned int BLOCK_SIZE = 256;
constexpr unsigned long long INVALID_INDEX = ~0ULL;

struct Candidate {
    float value;
    unsigned long long index;
};

__device__ __forceinline__ Candidate selectBetter(Candidate current, Candidate candidate) {
    if (candidate.index == INVALID_INDEX) {
        return current;
    }
    if (current.index == INVALID_INDEX) {
        return candidate;
    }

    const bool current_is_nan = current.value != current.value;
    const bool candidate_is_nan = candidate.value != candidate.value;
    if (candidate_is_nan != current_is_nan) {
        return candidate_is_nan ? candidate : current;
    }
    if (candidate_is_nan || candidate.value == current.value) {
        return candidate.index < current.index ? candidate : current;
    }
    return candidate.value > current.value ? candidate : current;
}

template <typename T>
__global__ void argmaxKernel(int64_t *max_index, T *max_value, const T *values, size_t count) {
    Candidate candidate{0.0f, INVALID_INDEX};
    for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
        candidate = selectBetter(candidate,
                                 Candidate{toFloat(values[i]), static_cast<unsigned long long>(i)});
    }

    __shared__ Candidate candidates[BLOCK_SIZE];
    candidates[threadIdx.x] = candidate;
    __syncthreads();

    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            candidates[threadIdx.x] = selectBetter(
                candidates[threadIdx.x], candidates[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        max_index[0] = static_cast<int64_t>(candidates[0].index);
        max_value[0] = values[candidates[0].index];
    }
}

template <typename T>
void launch(int64_t *max_index, std::byte *max_value, const std::byte *values, size_t count) {
    argmaxKernel<<<1, BLOCK_SIZE>>>(max_index, reinterpret_cast<T *>(max_value),
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
} // namespace llaisys::ops::cuda
