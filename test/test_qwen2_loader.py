import gc
import json
import tempfile
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

import llaisys


def test_qwen2_loader():
    config = {
        "torch_dtype": "float32",
        "num_hidden_layers": 1,
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 8,
        "max_position_embeddings": 16,
        "vocab_size": 10,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "eos_token_id": 9,
    }
    weights = {
        "model.embed_tokens.weight": np.zeros((10, 4), dtype=np.float32),
        "lm_head.weight": np.zeros((10, 4), dtype=np.float32),
        "model.norm.weight": np.ones((4,), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.ones((4,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.bias": np.zeros((4,), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros((2, 4), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.bias": np.zeros((2,), dtype=np.float32),
        "model.layers.0.self_attn.v_proj.weight": np.zeros((2, 4), dtype=np.float32),
        "model.layers.0.self_attn.v_proj.bias": np.zeros((2,), dtype=np.float32),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.ones((4,), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((8, 4), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((8, 4), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((4, 8), dtype=np.float32),
    }

    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory)
        (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        save_file(weights, model_path / "model.safetensors")
        model = llaisys.models.Qwen2(model_path)
        del model
        gc.collect()


if __name__ == "__main__":
    test_qwen2_loader()
    print("\n\033[92mTest passed!\033[0m\n")
