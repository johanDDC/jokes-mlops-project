import re
from pathlib import Path

import dvc.api as dvc
import hydra
from configs.experiment_cfg import ExperimentConfig
from hydra.core.config_store import ConfigStore

from jokes_mlops_project import NGramLanguageModel, Tokenizer, model_to_json


cs = ConfigStore.instance()
cs.store(name="run1", node=ExperimentConfig)


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

    # get data from dvc
    data: str = dvc.read(str(data_dir / data_file), repo=repository_path, mode="r")
    data = data.strip().replace("<|startoftext|>", "").split("\n\n")
    data = preprocess_text(data)

    # train model
    tokenizer = Tokenizer(**cfg.tokenizer_cfg)
    language_model = NGramLanguageModel(tokenizer, **cfg.model_cfg)
    language_model.fit(data)

    # save model
    save_path.mkdir(parents=True, exist_ok=True)
    save_path /= "model.json"
    with save_path.open("w") as outfile:
        model_to_json(language_model, outfile)


if __name__ == "__main__":
    main()
