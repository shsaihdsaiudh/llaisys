#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {
void add(std::byte *out, const std::byte *a, const std::byte *b, llaisysDataType_t dtype, size_t count);
}
