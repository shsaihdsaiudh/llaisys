#include "context.hpp"
#include "../../utils.hpp"
#include <thread>
#include <utility>

namespace llaisys::core {

Context::Context() : _current_runtime(nullptr) {
    // All device types, put CPU at the end
    std::vector<llaisysDeviceType_t> device_types;
    for (int i = 1; i < LLAISYS_DEVICE_TYPE_COUNT; i++) {
        device_types.push_back(static_cast<llaisysDeviceType_t>(i));
    }
    device_types.push_back(LLAISYS_DEVICE_CPU);

    // Create runtimes for each device type.
    // Activate the first available device. If no other device is available, activate CPU runtime.
    for (auto device_type : device_types) {
        const LlaisysRuntimeAPI *api = llaisysGetRuntimeAPI(device_type);
        const int device_count = api->get_device_count();
        CHECK_ARGUMENT(device_count >= 0, "runtime returned a negative device count");
        std::vector<runtime_ptr> runtimes_(static_cast<size_t>(device_count));
        if (_current_runtime == nullptr && device_count > 0) {
            auto runtime = runtime_ptr(new Runtime(device_type, 0));
            runtime->_activate();
            runtimes_[0] = runtime;
            _current_runtime = std::move(runtime);
        }
        _runtime_map.emplace(device_type, std::move(runtimes_));
    }
}

void Context::setDevice(llaisysDeviceType_t device_type, int device_id) {
    const auto runtime_entry = _runtime_map.find(device_type);
    CHECK_ARGUMENT(runtime_entry != _runtime_map.end(), "invalid device type");
    auto &runtimes = runtime_entry->second;
    CHECK_ARGUMENT(device_id >= 0 && static_cast<size_t>(device_id) < runtimes.size(), "invalid device id");

    auto &target = runtimes[static_cast<size_t>(device_id)];
    if (target == nullptr) {
        target = runtime_ptr(new Runtime(device_type, device_id));
    }
    target->_activate();
    _current_runtime = target;
}

Runtime &Context::runtime() {
    ASSERT(_current_runtime != nullptr, "No runtime is activated, please call setDevice() first.");
    _current_runtime->_activate();
    return *_current_runtime;
}

// Global API to get thread-local context.
Context &context() {
    thread_local Context thread_context;
    return thread_context;
}

} // namespace llaisys::core
