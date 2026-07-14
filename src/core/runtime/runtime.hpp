#pragma once
#include "../core.hpp"

#include "../../device/runtime_api.hpp"
#include "../allocator/allocator.hpp"

#include <memory>

namespace llaisys::core {
class Runtime : public std::enable_shared_from_this<Runtime> {
private:
    llaisysDeviceType_t _device_type;
    int _device_id;
    const LlaisysRuntimeAPI *_api;
    std::unique_ptr<MemoryAllocator> _allocator;
    void _activate();
    llaisysStream_t _stream;
    Runtime(llaisysDeviceType_t device_type, int device_id);

public:
    friend class Context;

    ~Runtime() noexcept;

    // Prevent copying
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    // Prevent moving
    Runtime(Runtime &&) = delete;
    Runtime &operator=(Runtime &&) = delete;

    llaisysDeviceType_t deviceType() const;
    int deviceId() const;

    const LlaisysRuntimeAPI *api() const;

    storage_t allocateDeviceStorage(size_t size);
    storage_t allocateHostStorage(size_t size);
    void freeStorage(Storage *storage) noexcept;

    llaisysStream_t stream() const;
    void synchronize() const;
};
} // namespace llaisys::core
