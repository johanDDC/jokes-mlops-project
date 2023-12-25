from dataclasses import dataclass, field
from typing import List

from .mlflow_cfg import MLFlowConfig
from .model_cfg import ModelConfig
from .tokenizer_cfg import TokenizerConfig


@dataclass
class ExperimentConfig:
    model_cfg: ModelConfig = ModelConfig(3)
    tokenizer_cfg: TokenizerConfig = TokenizerConfig(50_000, 6, 0.95)
    mlflow_cfg: MLFlowConfig = MLFlowConfig("ngram_lang_model")

    prefixes: List[str] = field(
        default_factory=lambda: [
            "заходит мужик в бар",
            "купил мужик шляпу",
            "идёт медведь по лесу",
        ]
    )
