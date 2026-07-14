#pragma once
#include "llaisys.h"

#include "../core.hpp"

#include <memory>

namespace llaisys::core {
class Storage {
private:
    std::byte *_memory;
    size_t _size;
    std::shared_ptr<Runtime> _runtime;
    bool _is_host;
    Storage(std::byte *memory, size_t size, std::shared_ptr<Runtime> runtime, bool is_host);

public:
    friend class Runtime;
    ~Storage();

    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;
    Storage(Storage &&) = delete;
    Storage &operator=(Storage &&) = delete;

    std::byte *memory() const;
    size_t size() const;
    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    bool isHost() const;
};

}; // namespace llaisys::core
