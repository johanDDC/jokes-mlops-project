# Hyperparameters tunning

This directory contains various configuration files for setting up the behavior
of the model. Specifically:

- `mlflow_cfg.py` provides the default host and port for the mlflow server. It
  also requires the mlflow run name.
- `tokenizer_cfg.py` provides the hyperparameters for the tokenizer:
  - `dictionary size` - the maximum number of tokens in the vocabulary;
  - `ngram_max` - the maximum number of subtokens that can be merged into a new
    token;
  - `pct_bpe` - the percentage of vocabulary size that will be filled using BPE
    training procedure.
- `model_cfg.py` provides a single hyperparameter: the number of worg N-grams
  for model.
- `experiment.cfg` provides a config for setting up the entire experiment. It
  includes the following fields:
  - `model_cfg`, `tokenizer_cfg` and `mlflow_cfg` - instances of base configs
    described above;
  - `git_commit_id` - link to the last up-to-date code version;
  - `prefixes` - list of prefixes that model should continue during inference
    time.
