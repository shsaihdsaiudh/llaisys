#include "argmax_nvidia.hpp"

#include "../../cuda_common.cuh"

namespace llaisys::ops::nvidia {
namespace {
constexpr unsigned int BLOCK_SIZE = 256;
constexpr unsigned int WARP_SIZE = 32;
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

__device__ __forceinline__ Candidate warpReduce(Candidate candidate) {
    const unsigned int lane = threadIdx.x & (WARP_SIZE - 1);
    constexpr unsigned int FULL_MASK = 0xffffffffu;
    for (unsigned int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        Candidate other;
        other.value = __shfl_down_sync(FULL_MASK, candidate.value, offset);
        other.index = __shfl_down_sync(FULL_MASK, candidate.index, offset);
        if (lane + offset < WARP_SIZE) {
            candidate = selectBetter(candidate, other);
        }
    }
    return candidate;
}

template <typename T>
__global__ void argmaxKernel(int64_t *max_index, T *max_value, const T *values, size_t count) {
    Candidate candidate{0.0f, INVALID_INDEX};
    for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
        candidate = selectBetter(candidate,
                                 Candidate{toFloat(values[i]), static_cast<unsigned long long>(i)});
    }

    candidate = warpReduce(candidate);

    __shared__ Candidate warp_candidates[BLOCK_SIZE / WARP_SIZE];
    const unsigned int lane = threadIdx.x & (WARP_SIZE - 1);
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        warp_candidates[warp] = candidate;
    }
    __syncthreads();

    if (warp == 0) {
        candidate = lane < BLOCK_SIZE / WARP_SIZE
                      ? warp_candidates[lane]
                      : Candidate{0.0f, INVALID_INDEX};
        candidate = warpReduce(candidate);
        if (lane == 0) {
            max_index[0] = static_cast<int64_t>(candidate.index);
            max_value[0] = values[candidate.index];
        }
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
} // namespace llaisys::ops::nvidia
