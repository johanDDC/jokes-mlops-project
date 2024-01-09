import json
from pathlib import Path

from jokes_mlops_project.model.model import NGramLanguageModel
from jokes_mlops_project.model.torenizer import Tokenizer


def serialize_probs(probs):
    new_probs = {}
    for key in probs.keys():
        new_probs[" ".join(key)] = probs[key]
    return new_probs


def deserialize_probs(probs):
    new_probs = {}
    for key in probs.keys():
        new_probs[tuple(key.split())] = probs[key]
    return new_probs


def model_to_json(model: NGramLanguageModel, outfile, **kwargs):
    probs = serialize_probs(model.probs)
    result_dict = {
        "n_grams": model.n_grams,
        "tokenizer": model.tokenizer.bpe.vocabs_to_dict(True),
        "probs": probs,
    }
    for key in kwargs.keys():
        result_dict.update({key: kwargs.get(key)})
    json.dump(result_dict, outfile)


def model_from_json(filepath: Path) -> NGramLanguageModel:
    with filepath.open("r") as file:
        obj = json.load(file)
    tokenizer = Tokenizer.load(obj["tokenizer"])
    model = NGramLanguageModel(tokenizer, obj["n_grams"])
    probs = deserialize_probs(obj["probs"])
    model.probs = probs
    del obj["tokenizer"], obj["n_grams"], obj["probs"]
    return model, obj
