#pragma once

#include "llaisys.h"

namespace llaisys::ops::cuda {
inline bool isAvailableDevice(llaisysDeviceType_t device_type) {
    switch (device_type) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return true;
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return true;
#endif
    default:
        return false;
    }
}
} // namespace llaisys::ops::cuda
