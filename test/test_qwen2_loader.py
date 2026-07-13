import gc
import json
import tempfile
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
import torch
import torch.nn.functional as F

import llaisys


def reference_next_token(token_ids, weights, config):
    tensors = {name: torch.from_numpy(value) for name, value in weights.items()}
    hidden = F.embedding(torch.tensor(token_ids), tensors["model.embed_tokens.weight"])
    head_dim = config["hidden_size"] // config["num_attention_heads"]

    def rms_norm(value, weight):
        variance = value.pow(2).mean(dim=-1, keepdim=True)
        return value * torch.rsqrt(variance + config["rms_norm_eps"]) * weight

    def rope(value):
        half = head_dim // 2
        positions = torch.arange(len(token_ids), dtype=torch.float32).unsqueeze(1)
        indices = torch.arange(half, dtype=torch.float32)
        angles = positions / (config["rope_theta"] ** (2 * indices / head_dim))
        sine = angles.sin().unsqueeze(1)
        cosine = angles.cos().unsqueeze(1)
        first, second = value[..., :half], value[..., half:]
        return torch.cat(
            (first * cosine - second * sine, second * cosine + first * sine),
            dim=-1,
        )

    prefix = "model.layers.0."
    residual = hidden
    normalized = rms_norm(hidden, tensors[prefix + "input_layernorm.weight"])
    query = F.linear(
        normalized,
        tensors[prefix + "self_attn.q_proj.weight"],
        tensors[prefix + "self_attn.q_proj.bias"],
    ).view(len(token_ids), config["num_attention_heads"], head_dim)
    key = F.linear(
        normalized,
        tensors[prefix + "self_attn.k_proj.weight"],
        tensors[prefix + "self_attn.k_proj.bias"],
    ).view(len(token_ids), config["num_key_value_heads"], head_dim)
    value = F.linear(
        normalized,
        tensors[prefix + "self_attn.v_proj.weight"],
        tensors[prefix + "self_attn.v_proj.bias"],
    ).view(len(token_ids), config["num_key_value_heads"], head_dim)
    query, key = rope(query), rope(key)

    query = query.transpose(0, 1)
    key = key.transpose(0, 1).repeat_interleave(
        config["num_attention_heads"] // config["num_key_value_heads"], dim=0
    )
    value = value.transpose(0, 1).repeat_interleave(
        config["num_attention_heads"] // config["num_key_value_heads"], dim=0
    )
    scores = query @ key.transpose(-2, -1) / (head_dim**0.5)
    mask = torch.ones(len(token_ids), len(token_ids), dtype=torch.bool).tril()
    scores.masked_fill_(~mask, float("-inf"))
    attention = (scores.softmax(dim=-1) @ value).transpose(0, 1).reshape(
        len(token_ids), config["hidden_size"]
    )
    hidden = residual + F.linear(attention, tensors[prefix + "self_attn.o_proj.weight"])

    residual = hidden
    normalized = rms_norm(
        hidden, tensors[prefix + "post_attention_layernorm.weight"]
    )
    gate = F.linear(normalized, tensors[prefix + "mlp.gate_proj.weight"])
    up = F.linear(normalized, tensors[prefix + "mlp.up_proj.weight"])
    hidden = residual + F.linear(
        F.silu(gate) * up, tensors[prefix + "mlp.down_proj.weight"]
    )
    hidden = rms_norm(hidden[-1:], tensors["model.norm.weight"])
    logits = F.linear(hidden, tensors["lm_head.weight"])
    return int(logits.argmax(dim=-1).item())


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
        assert model.generate([1, 2], max_new_tokens=2, top_k=1) == [1, 2, 0, 0]
        del model
        gc.collect()


def test_qwen2_reference_generation():
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
        "eos_token_id": -1,
    }
    rng = np.random.default_rng(7)

    def random(shape, scale=0.2):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    weights = {
        "model.embed_tokens.weight": random((10, 4)),
        "lm_head.weight": random((10, 4)),
        "model.norm.weight": 1.0 + random((4,), 0.05),
        "model.layers.0.input_layernorm.weight": 1.0 + random((4,), 0.05),
        "model.layers.0.self_attn.q_proj.weight": random((4, 4)),
        "model.layers.0.self_attn.q_proj.bias": random((4,)),
        "model.layers.0.self_attn.k_proj.weight": random((2, 4)),
        "model.layers.0.self_attn.k_proj.bias": random((2,)),
        "model.layers.0.self_attn.v_proj.weight": random((2, 4)),
        "model.layers.0.self_attn.v_proj.bias": random((2,)),
        "model.layers.0.self_attn.o_proj.weight": random((4, 4)),
        "model.layers.0.post_attention_layernorm.weight": 1.0
        + random((4,), 0.05),
        "model.layers.0.mlp.gate_proj.weight": random((8, 4)),
        "model.layers.0.mlp.up_proj.weight": random((8, 4)),
        "model.layers.0.mlp.down_proj.weight": random((4, 8)),
    }
    expected = [1, 2]
    for _ in range(2):
        expected.append(reference_next_token(expected, weights, config))

    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory)
        (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        save_file(weights, model_path / "model.safetensors")
        model = llaisys.models.Qwen2(model_path)
        assert model.generate([1, 2], max_new_tokens=2, top_k=1) == expected
        del model
        gc.collect()


if __name__ == "__main__":
    test_qwen2_loader()
    test_qwen2_reference_generation()
    print("\n\033[92mTest passed!\033[0m\n")
