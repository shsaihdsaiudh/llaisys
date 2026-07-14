#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::nvidia {
void argmax(int64_t *max_index, std::byte *max_value, const std::byte *values,
            llaisysDataType_t dtype, size_t count);
}
