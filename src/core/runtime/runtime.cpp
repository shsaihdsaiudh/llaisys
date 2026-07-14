#include "runtime.hpp"

#include "../../device/runtime_api.hpp"
#include "../allocator/naive_allocator.hpp"

#include <exception>
#include <iostream>

namespace llaisys::core {
Runtime::Runtime(llaisysDeviceType_t device_type, int device_id)
    : _device_type(device_type),
      _device_id(device_id),
      _api(llaisys::device::getRuntimeAPI(device_type)),
      _allocator(std::make_unique<allocators::NaiveAllocator>(_api)),
      _stream(nullptr) {
    _api->set_device(_device_id);
    _stream = _api->create_stream();
}

Runtime::~Runtime() noexcept {
    try {
        _api->set_device(_device_id);
    } catch (const std::exception &error) {
        std::cerr << "Failed to select device while destroying runtime: " << error.what() << std::endl;
        static_cast<void>(_allocator.release());
        _stream = nullptr;
        _api = nullptr;
        return;
    } catch (...) {
        std::cerr << "Failed to select device while destroying runtime: unknown error" << std::endl;
        static_cast<void>(_allocator.release());
        _stream = nullptr;
        _api = nullptr;
        return;
    }

    _allocator.reset();
    try {
        _api->destroy_stream(_stream);
    } catch (const std::exception &error) {
        std::cerr << "Failed to destroy runtime stream: " << error.what() << std::endl;
    } catch (...) {
        std::cerr << "Failed to destroy runtime stream: unknown error" << std::endl;
    }
    _stream = nullptr;
    _api = nullptr;
}

void Runtime::_activate() {
    _api->set_device(_device_id);
}

llaisysDeviceType_t Runtime::deviceType() const {
    return _device_type;
}

int Runtime::deviceId() const {
    return _device_id;
}

const LlaisysRuntimeAPI *Runtime::api() const {
    return _api;
}

storage_t Runtime::allocateDeviceStorage(size_t size) {
    _api->set_device(_device_id);
    return std::shared_ptr<Storage>(new Storage(_allocator->allocate(size), size, shared_from_this(), false));
}

storage_t Runtime::allocateHostStorage(size_t size) {
    _api->set_device(_device_id);
    return std::shared_ptr<Storage>(new Storage(
        static_cast<std::byte *>(_api->malloc_host(size)), size, shared_from_this(), true));
}

void Runtime::freeStorage(Storage *storage) noexcept {
    try {
        _api->set_device(_device_id);
        if (storage->isHost()) {
            _api->free_host(storage->memory());
        } else {
            _allocator->release(storage->memory());
        }
    } catch (const std::exception &error) {
        std::cerr << "Failed to release storage: " << error.what() << std::endl;
    } catch (...) {
        std::cerr << "Failed to release storage: unknown error" << std::endl;
    }
}

llaisysStream_t Runtime::stream() const {
    return _stream;
}

void Runtime::synchronize() const {
    _api->set_device(_device_id);
    _api->stream_synchronize(_stream);
}

} // namespace llaisys::core
