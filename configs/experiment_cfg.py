from dataclasses import dataclass

from .model_cfg import ModelConfig
from .tokenizer_cfg import TokenizerConfig


@dataclass
class ExperimentConfig:
    model_cfg: ModelConfig = ModelConfig(3)
    tokenizer_cfg: TokenizerConfig = TokenizerConfig(50_000, 6, 0.95)
