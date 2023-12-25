from dataclasses import dataclass


@dataclass
class MLFlowConfig:
    experiment_name: str
    host: str = "128.0.1.1"
    port: int = 8080
