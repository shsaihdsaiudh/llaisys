#include "../runtime_api.hpp"

#include <mcr/mc_runtime.h>

#include <sstream>
#include <stdexcept>

namespace llaisys::device::metax {
namespace runtime_api {
namespace {
void checkMaca(mcError_t status, const char *operation) {
    if (status == mcSuccess) {
        return;
    }
    std::ostringstream message;
    message << operation << " failed: " << mcGetErrorString(status);
    throw std::runtime_error(message.str());
}

mcMemcpyKind memcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return mcMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return mcMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return mcMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return mcMemcpyDeviceToDevice;
    default:
        throw std::invalid_argument("Unsupported MACA memcpy kind");
    }
}
} // namespace

int getDeviceCount() {
    int count = 0;
    checkMaca(mcGetDeviceCount(&count), "mcGetDeviceCount");
    return count;
}

void setDevice(int device) {
    checkMaca(mcSetDevice(device), "mcSetDevice");
}

void deviceSynchronize() {
    checkMaca(mcDeviceSynchronize(), "mcDeviceSynchronize");
}

llaisysStream_t createStream() {
    mcStream_t stream = nullptr;
    checkMaca(mcStreamCreate(&stream), "mcStreamCreate");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream != nullptr) {
        checkMaca(mcStreamDestroy(reinterpret_cast<mcStream_t>(stream)), "mcStreamDestroy");
    }
}

void streamSynchronize(llaisysStream_t stream) {
    checkMaca(mcStreamSynchronize(reinterpret_cast<mcStream_t>(stream)), "mcStreamSynchronize");
}

void *mallocDevice(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    checkMaca(mcMalloc(&ptr, size), "mcMalloc");
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr != nullptr) {
        checkMaca(mcFree(ptr), "mcFree");
    }
}

void *mallocHost(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    checkMaca(mcMallocHost(&ptr, size), "mcMallocHost");
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr != nullptr) {
        checkMaca(mcFreeHost(ptr), "mcFreeHost");
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    if (size != 0) {
        checkMaca(mcMemcpy(dst, src, size, memcpyKind(kind)), "mcMemcpy");
    }
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind,
                 llaisysStream_t stream) {
    if (size != 0) {
        checkMaca(mcMemcpyAsync(dst, src, size, memcpyKind(kind),
                                reinterpret_cast<mcStream_t>(stream)),
                  "mcMemcpyAsync");
    }
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};
} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::metax
