from dataclasses import dataclass


@dataclass
class MLFlowConfig:
    experiment_name: str
    host: str = "127.0.0.1"
    port: int = 5000
