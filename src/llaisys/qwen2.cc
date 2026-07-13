#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"

#include "../tensor/tensor.hpp"
#include "../utils.hpp"

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

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *, int64_t *, size_t) {
        TO_BE_IMPLEMENTED();
    }
}
