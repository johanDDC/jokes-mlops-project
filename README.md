# jokes-mlops-project

## Description

The `jokes-mlops-project` module implements the `NGramLanguageModel` class,
which uses an N-gram language model to generate jokes. The text is tokenized
using BPE before being fed into the model.

## Running a model

All main hyperparameters may be setuped in files `configs/*_cfg.py`. See
[`configs/README.md`](./configs/README.md) for details. After you are done with
setups, the train/infer process may be run using one of the following commands:

- `python train.py` for training;
- `python infer.py` for inference;

## Data

Model is trained using jokes parsed from `https://www.anekdot.ru/`.

## License

This project is licensed under the terms of the MIT license. See the `LICENSE`
file for more information.
