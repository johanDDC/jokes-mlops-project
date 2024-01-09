import re
from pathlib import Path

import dvc.api as dvc
import hydra
import mlflow
import numpy as np
from configs.experiment_cfg import ExperimentConfig
from hydra.core.config_store import ConfigStore

from jokes_mlops_project import NGramLanguageModel, Tokenizer, model_to_json, perplexity
from jokes_mlops_project.utils.utils import Timer


def preprocess_text(text):
    punkt = "-"
    new_text = []
    for line in text:
        new_line = line.lower()
        new_line = re.sub(punkt, "", new_line)
        new_line = new_line.replace("...", " ")
        new_text.append(new_line.strip())
    return new_text


@hydra.main(version_base=None, config_name="run1")
def main(cfg: ExperimentConfig):
    repository_path = "https://github.com/johanDDC/jokes-mlops-project.git"
    data_dir = Path("data")
    data_file = "data.txt"
    save_path = Path("chekpoints")
    timer = Timer()

    # get data from dvc
    data: str = dvc.read(str(data_dir / data_file), repo=repository_path, mode="r")
    data = data.strip().replace("<|startoftext|>", "").split("\n\n")
    data = preprocess_text(data)

    # setup mlflow
    mlflow.set_tracking_uri(uri=f"http://{cfg.mlflow_cfg.host}:{cfg.mlflow_cfg.port}")
    # try:
    exp_id = mlflow.create_experiment(cfg.mlflow_cfg.experiment_name)
    mlflow.set_experiment(cfg.mlflow_cfg.experiment_name)
    with mlflow.start_run(run_name="train", experiment_id=exp_id) as parent_run:
        run_id = parent_run.info.run_id
        # with mlflow.start_run(run_name=f"train", experiment_id=exp_id, nested=True) as child_run:
        params = dict(cfg)
        mlflow.log_params(params)
        # train model
        tokenizer = Tokenizer(**cfg.tokenizer_cfg)
        language_model = NGramLanguageModel(tokenizer, **cfg.model_cfg)
        with timer:
            language_model.fit(data)

        model_train_time = timer.time
        mlflow.log_metric("model_train_time", model_train_time)

        # count perplexity
        test_ids = np.random.randint(0, len(data), int(0.01 * len(data)))
        ppl = perplexity(language_model, data, test_ids)
        mlflow.log_metric("perplexity", ppl)

    # save model
    save_path.mkdir(parents=True, exist_ok=True)
    save_path /= "model.json"
    with save_path.open("w") as outfile:
        model_to_json(language_model, outfile, exp_id=exp_id, run_id=run_id)


if __name__ == "__main__":
    config_store = ConfigStore.instance()
    config_store.store(name="run1", node=ExperimentConfig)

    main()
