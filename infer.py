from pathlib import Path

import hydra
import mlflow
from configs.experiment_cfg import ExperimentConfig
from hydra.core.config_store import ConfigStore

from jokes_mlops_project import model_from_json


@hydra.main(version_base=None, config_name="run1")
def main(cfg: ExperimentConfig):
    # load model
    model_path = Path("chekpoints") / "model.json"
    language_model, args = model_from_json(model_path)

    mlflow.set_tracking_uri(uri=f"http://{cfg.mlflow_cfg.host}:{cfg.mlflow_cfg.port}")
    mlflow.set_experiment(cfg.mlflow_cfg.experiment_name)
    # with mlflow.start_run(run_id=args["run_id"]) as parent_run:
    with mlflow.start_run(run_name="infer", experiment_id=args["exp_id"]):
        # generate
        prefixes = cfg.prefixes
        for i, prefix in enumerate(prefixes):
            mlflow.log_text(language_model.generate(prefix), f"generate_{i+1}.txt")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="run1", node=ExperimentConfig)

    main()
