#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"

#include "../tensor/tensor.hpp"
#include "../utils.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device;
    int device_id;
    LlaisysQwen2Weights weights{};
    std::vector<llaisys::tensor_t> key_cache;
    std::vector<llaisys::tensor_t> value_cache;
    size_t cache_length = 0;
};

namespace {
llaisysTensor_t make_tensor(const std::vector<size_t> &shape,
                            llaisysDataType_t dtype,
                            llaisysDeviceType_t device,
                            int device_id) {
    return new LlaisysTensor{llaisys::Tensor::create(shape, dtype, device, device_id)};
}

void destroy_tensor(llaisysTensor_t tensor) {
    delete tensor;
}

llaisysTensor_t *make_layer_array(size_t layers) {
    return new llaisysTensor_t[layers]{};
}

void destroy_layer_array(llaisysTensor_t *tensors, size_t layers) {
    if (tensors == nullptr) {
        return;
    }
    for (size_t layer = 0; layer < layers; ++layer) {
        destroy_tensor(tensors[layer]);
    }
    delete[] tensors;
}

llaisys::tensor_t tensor(llaisysTensor_t handle) {
    return handle->tensor;
}
} // namespace

__C {
    LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                               llaisysDeviceType_t device,
                                               int *device_ids,
                                               int ndevice) {
        CHECK_ARGUMENT(meta != nullptr, "Qwen2 metadata must not be null");
        CHECK_ARGUMENT(device == LLAISYS_DEVICE_CPU, "Qwen2 currently supports CPU inference only");
        CHECK_ARGUMENT(ndevice == 1 && device_ids != nullptr, "Qwen2 requires exactly one device");
        CHECK_ARGUMENT(meta->nlayer > 0 && meta->hs > 0 && meta->nh > 0 && meta->nkvh > 0,
                       "Qwen2 dimensions must be positive");
        CHECK_ARGUMENT(meta->hs == meta->nh * meta->dh, "Qwen2 hidden size must equal heads times head dimension");
        CHECK_ARGUMENT(meta->nh % meta->nkvh == 0, "Qwen2 query heads must be divisible by KV heads");
        CHECK_ARGUMENT(meta->di > 0 && meta->maxseq > 0 && meta->voc > 0,
                       "Qwen2 intermediate, sequence, and vocabulary sizes must be positive");

        auto *model = new LlaisysQwen2Model{*meta, device, device_ids[0]};
        auto &weights = model->weights;

        weights.in_embed = make_tensor({meta->voc, meta->hs}, meta->dtype, device, model->device_id);
        weights.out_embed = make_tensor({meta->voc, meta->hs}, meta->dtype, device, model->device_id);
        weights.out_norm_w = make_tensor({meta->hs}, meta->dtype, device, model->device_id);

        weights.attn_norm_w = make_layer_array(meta->nlayer);
        weights.attn_q_w = make_layer_array(meta->nlayer);
        weights.attn_q_b = make_layer_array(meta->nlayer);
        weights.attn_k_w = make_layer_array(meta->nlayer);
        weights.attn_k_b = make_layer_array(meta->nlayer);
        weights.attn_v_w = make_layer_array(meta->nlayer);
        weights.attn_v_b = make_layer_array(meta->nlayer);
        weights.attn_o_w = make_layer_array(meta->nlayer);
        weights.mlp_norm_w = make_layer_array(meta->nlayer);
        weights.mlp_gate_w = make_layer_array(meta->nlayer);
        weights.mlp_up_w = make_layer_array(meta->nlayer);
        weights.mlp_down_w = make_layer_array(meta->nlayer);

        for (size_t layer = 0; layer < meta->nlayer; ++layer) {
            weights.attn_norm_w[layer] = make_tensor({meta->hs}, meta->dtype, device, model->device_id);
            weights.attn_q_w[layer] = make_tensor({meta->nh * meta->dh, meta->hs}, meta->dtype, device, model->device_id);
            weights.attn_q_b[layer] = make_tensor({meta->nh * meta->dh}, meta->dtype, device, model->device_id);
            weights.attn_k_w[layer] = make_tensor({meta->nkvh * meta->dh, meta->hs}, meta->dtype, device, model->device_id);
            weights.attn_k_b[layer] = make_tensor({meta->nkvh * meta->dh}, meta->dtype, device, model->device_id);
            weights.attn_v_w[layer] = make_tensor({meta->nkvh * meta->dh, meta->hs}, meta->dtype, device, model->device_id);
            weights.attn_v_b[layer] = make_tensor({meta->nkvh * meta->dh}, meta->dtype, device, model->device_id);
            weights.attn_o_w[layer] = make_tensor({meta->hs, meta->hs}, meta->dtype, device, model->device_id);
            weights.mlp_norm_w[layer] = make_tensor({meta->hs}, meta->dtype, device, model->device_id);
            weights.mlp_gate_w[layer] = make_tensor({meta->di, meta->hs}, meta->dtype, device, model->device_id);
            weights.mlp_up_w[layer] = make_tensor({meta->di, meta->hs}, meta->dtype, device, model->device_id);
            weights.mlp_down_w[layer] = make_tensor({meta->hs, meta->di}, meta->dtype, device, model->device_id);
        }

        model->key_cache.reserve(meta->nlayer);
        model->value_cache.reserve(meta->nlayer);
        for (size_t layer = 0; layer < meta->nlayer; ++layer) {
            model->key_cache.push_back(llaisys::Tensor::create(
                {meta->maxseq, meta->nkvh, meta->dh}, meta->dtype, device, model->device_id));
            model->value_cache.push_back(llaisys::Tensor::create(
                {meta->maxseq, meta->nkvh, meta->dh}, meta->dtype, device, model->device_id));
        }

        return model;
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model * model) {
        if (model == nullptr) {
            return;
        }
        auto &weights = model->weights;
        destroy_tensor(weights.in_embed);
        destroy_tensor(weights.out_embed);
        destroy_tensor(weights.out_norm_w);
        destroy_layer_array(weights.attn_norm_w, model->meta.nlayer);
        destroy_layer_array(weights.attn_q_w, model->meta.nlayer);
        destroy_layer_array(weights.attn_q_b, model->meta.nlayer);
        destroy_layer_array(weights.attn_k_w, model->meta.nlayer);
        destroy_layer_array(weights.attn_k_b, model->meta.nlayer);
        destroy_layer_array(weights.attn_v_w, model->meta.nlayer);
        destroy_layer_array(weights.attn_v_b, model->meta.nlayer);
        destroy_layer_array(weights.attn_o_w, model->meta.nlayer);
        destroy_layer_array(weights.mlp_norm_w, model->meta.nlayer);
        destroy_layer_array(weights.mlp_gate_w, model->meta.nlayer);
        destroy_layer_array(weights.mlp_up_w, model->meta.nlayer);
        destroy_layer_array(weights.mlp_down_w, model->meta.nlayer);
        delete model;
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model * model) {
        CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
        return &model->weights;
    }

    void llaisysQwen2ModelReset(LlaisysQwen2Model * model) {
        CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
        model->cache_length = 0;
    }

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
        CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
        CHECK_ARGUMENT(token_ids != nullptr && ntoken > 0, "Qwen2 inference requires at least one token");
        CHECK_ARGUMENT(model->cache_length + ntoken <= model->meta.maxseq,
                       "Qwen2 sequence exceeds the configured maximum length");
        for (size_t i = 0; i < ntoken; ++i) {
            CHECK_ARGUMENT(token_ids[i] >= 0 && static_cast<size_t>(token_ids[i]) < model->meta.voc,
                           "Qwen2 token ID is out of range");
        }

        const auto &meta = model->meta;
        auto &weights = model->weights;
        auto tokens = llaisys::Tensor::create({ntoken}, LLAISYS_DTYPE_I64, model->device, model->device_id);
        tokens->load(token_ids);
        auto hidden = llaisys::Tensor::create({ntoken, meta.hs}, meta.dtype, model->device, model->device_id);
        llaisys::ops::embedding(hidden, tokens, tensor(weights.in_embed));

        std::vector<int64_t> position_values(ntoken);
        for (size_t i = 0; i < ntoken; ++i) {
            position_values[i] = static_cast<int64_t>(model->cache_length + i);
        }
        auto positions = llaisys::Tensor::create({ntoken}, LLAISYS_DTYPE_I64, model->device, model->device_id);
        positions->load(position_values.data());

        const size_t total_length = model->cache_length + ntoken;
        const size_t kv_width = meta.nkvh * meta.dh;
        const size_t element_size = llaisys::utils::dsize(meta.dtype);
        const size_t cache_copy_bytes = ntoken * kv_width * element_size;

        for (size_t layer = 0; layer < meta.nlayer; ++layer) {
            auto normalized = llaisys::Tensor::create({ntoken, meta.hs}, meta.dtype, model->device, model->device_id);
            llaisys::ops::rms_norm(normalized, hidden, tensor(weights.attn_norm_w[layer]), meta.epsilon);

            auto query = llaisys::Tensor::create({ntoken, meta.nh * meta.dh}, meta.dtype, model->device, model->device_id);
            auto key = llaisys::Tensor::create({ntoken, kv_width}, meta.dtype, model->device, model->device_id);
            auto value = llaisys::Tensor::create({ntoken, kv_width}, meta.dtype, model->device, model->device_id);
            llaisys::ops::linear(query, normalized, tensor(weights.attn_q_w[layer]), tensor(weights.attn_q_b[layer]));
            llaisys::ops::linear(key, normalized, tensor(weights.attn_k_w[layer]), tensor(weights.attn_k_b[layer]));
            llaisys::ops::linear(value, normalized, tensor(weights.attn_v_w[layer]), tensor(weights.attn_v_b[layer]));

            query = query->view({ntoken, meta.nh, meta.dh});
            key = key->view({ntoken, meta.nkvh, meta.dh});
            value = value->view({ntoken, meta.nkvh, meta.dh});
            llaisys::ops::rope(query, query, positions, meta.theta);
            llaisys::ops::rope(key, key, positions, meta.theta);

            const size_t cache_offset = model->cache_length * kv_width * element_size;
            std::memcpy(model->key_cache[layer]->data() + cache_offset, key->data(), cache_copy_bytes);
            std::memcpy(model->value_cache[layer]->data() + cache_offset, value->data(), cache_copy_bytes);
            auto cached_keys = model->key_cache[layer]->slice(0, 0, total_length);
            auto cached_values = model->value_cache[layer]->slice(0, 0, total_length);

            auto attention = llaisys::Tensor::create(
                {ntoken, meta.nh, meta.dh}, meta.dtype, model->device, model->device_id);
            llaisys::ops::self_attention(
                attention, query, cached_keys, cached_values, 1.0F / std::sqrt(static_cast<float>(meta.dh)));
            auto projected = llaisys::Tensor::create({ntoken, meta.hs}, meta.dtype, model->device, model->device_id);
            llaisys::ops::linear(
                projected, attention->view({ntoken, meta.hs}), tensor(weights.attn_o_w[layer]), nullptr);
            llaisys::ops::add(hidden, hidden, projected);

            normalized = llaisys::Tensor::create({ntoken, meta.hs}, meta.dtype, model->device, model->device_id);
            llaisys::ops::rms_norm(normalized, hidden, tensor(weights.mlp_norm_w[layer]), meta.epsilon);
            auto gate = llaisys::Tensor::create({ntoken, meta.di}, meta.dtype, model->device, model->device_id);
            auto up = llaisys::Tensor::create({ntoken, meta.di}, meta.dtype, model->device, model->device_id);
            llaisys::ops::linear(gate, normalized, tensor(weights.mlp_gate_w[layer]), nullptr);
            llaisys::ops::linear(up, normalized, tensor(weights.mlp_up_w[layer]), nullptr);
            llaisys::ops::swiglu(gate, gate, up);
            auto down = llaisys::Tensor::create({ntoken, meta.hs}, meta.dtype, model->device, model->device_id);
            llaisys::ops::linear(down, gate, tensor(weights.mlp_down_w[layer]), nullptr);
            llaisys::ops::add(hidden, hidden, down);
        }

        model->cache_length = total_length;
        auto last_hidden = hidden->slice(0, ntoken - 1, ntoken);
        auto normalized = llaisys::Tensor::create({1, meta.hs}, meta.dtype, model->device, model->device_id);
        llaisys::ops::rms_norm(normalized, last_hidden, tensor(weights.out_norm_w), meta.epsilon);
        auto logits = llaisys::Tensor::create({1, meta.voc}, meta.dtype, model->device, model->device_id);
        llaisys::ops::linear(logits, normalized, tensor(weights.out_embed), nullptr);
        auto max_index = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, model->device, model->device_id);
        auto max_value = llaisys::Tensor::create({1}, meta.dtype, model->device, model->device_id);
        llaisys::ops::argmax(max_index, max_value, logits->view({meta.voc}));
        return *reinterpret_cast<const int64_t *>(max_index->data());
    }
}
