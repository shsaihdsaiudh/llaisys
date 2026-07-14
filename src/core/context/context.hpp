#pragma once

#include "llaisys.h"

#include "../core.hpp"

#include "../runtime/runtime.hpp"

#include <unordered_map>
#include <vector>

namespace llaisys::core {
class Context {
private:
    using runtime_ptr = std::shared_ptr<Runtime>;

    std::unordered_map<llaisysDeviceType_t, std::vector<runtime_ptr>> _runtime_map;
    runtime_ptr _current_runtime;
    Context();

public:
    ~Context() = default;

    // Prevent copy
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    // Prevent move
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;

    void setDevice(llaisysDeviceType_t device_type, int device_id);
    Runtime &runtime();

    friend Context &context();
};
} // namespace llaisys::core
