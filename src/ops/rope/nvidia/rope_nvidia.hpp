#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cuda {
void rope(std::byte *out, const std::byte *input, const int64_t *positions,
          llaisysDataType_t dtype, size_t sequence_length, size_t heads,
          size_t head_dimension, float theta);
}
