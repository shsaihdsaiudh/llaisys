#include "linear_nvidia.hpp"

#include "../../cuda_common.cuh"

#include <cublas_v2.h>

#include <limits>
#include <sstream>

namespace llaisys::ops::cuda {
namespace {
void checkCublas(cublasStatus_t status, const char *operation) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    }
    std::ostringstream message;
    message << operation << " failed with cuBLAS status " << static_cast<int>(status);
    throw std::runtime_error(message.str());
}

cublasHandle_t handle() {
    thread_local cublasHandle_t value = [] {
        cublasHandle_t result = nullptr;
        checkCublas(cublasCreate(&result), "cublasCreate");
        checkCublas(cublasSetMathMode(result, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
        return result;
    }();
    return value;
}

cudaDataType_t cudaType(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return CUDA_R_32F;
    case LLAISYS_DTYPE_F16:
        return CUDA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return CUDA_R_16BF;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

template <typename T>
__global__ void addBiasKernel(T *out, const T *bias, size_t rows, size_t columns) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * columns) {
        out[index] = fromFloat<T>(toFloat(out[index]) + toFloat(bias[index % columns]));
    }
}

template <typename T>
void addBias(std::byte *out, const std::byte *bias, size_t rows, size_t columns) {
    constexpr int threads = 256;
    const size_t count = rows * columns;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    addBiasKernel<<<blocks, threads>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(bias),
                                      rows, columns);
    checkKernel("linear bias kernel");
}
} // namespace

void linear(std::byte *out, const std::byte *input, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t dtype,
            size_t rows, size_t input_features, size_t output_features) {
    CHECK_ARGUMENT(rows <= static_cast<size_t>(std::numeric_limits<int>::max())
                       && input_features <= static_cast<size_t>(std::numeric_limits<int>::max())
                       && output_features <= static_cast<size_t>(std::numeric_limits<int>::max()),
                   "CUDA linear dimensions exceed cuBLAS limits");
    if (rows == 0 || input_features == 0 || output_features == 0) {
        return;
    }

    const float alpha = 1.0F;
    const float beta = 0.0F;
    const auto type = cudaType(dtype);
    checkCublas(cublasGemmEx(
                    handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                    static_cast<int>(output_features), static_cast<int>(rows),
                    static_cast<int>(input_features), &alpha,
                    weight, type, static_cast<int>(input_features),
                    input, type, static_cast<int>(input_features), &beta,
                    out, type, static_cast<int>(output_features),
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                "cublasGemmEx");

    if (bias == nullptr) {
        return;
    }
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return addBias<float>(out, bias, rows, output_features);
    case LLAISYS_DTYPE_F16:
        return addBias<__half>(out, bias, rows, output_features);
    case LLAISYS_DTYPE_BF16:
        return addBias<__nv_bfloat16>(out, bias, rows, output_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cuda
