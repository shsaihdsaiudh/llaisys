#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {
void linear(std::byte *out, const std::byte *input, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t dtype,
            size_t rows, size_t input_features, size_t output_features);
}
