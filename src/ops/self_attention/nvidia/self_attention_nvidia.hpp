#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {
void selfAttention(std::byte *out, const std::byte *query, const std::byte *key,
                   const std::byte *value, llaisysDataType_t dtype,
                   size_t query_length, size_t kv_length, size_t query_heads,
                   size_t kv_heads, size_t head_dimension, size_t value_dimension,
                   float scale);
}
