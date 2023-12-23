from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    dictionary_size: int
    ngram_max: int
    pct_bpe: float
