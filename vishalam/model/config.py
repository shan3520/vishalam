from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 50304
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    ffn_dim: int = 2816
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def model_config_from_yaml(path: str | Path) -> ModelConfig:
    return ModelConfig(**load_yaml(path)["model"])
