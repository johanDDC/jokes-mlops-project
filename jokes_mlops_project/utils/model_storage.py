import json
from pathlib import Path

from jokes_mlops_project.model.model import NGramLanguageModel
from jokes_mlops_project.model.torenizer import Tokenizer


def serialize_probs(probs):
    new_probs = {}
    for k in probs.keys():
        new_probs[" ".join(k)] = probs[k]
    return new_probs


def deserialize_probs(probs):
    new_probs = {}
    for k in probs.keys():
        new_probs[tuple(k.split())] = probs[k]
    return new_probs


def model_to_json(model: NGramLanguageModel, outfile, **kwargs):
    probs = serialize_probs(model.probs)
    result_dict = {
        "n": model.n,
        "tokenizer": model.tokenizer.bpe.vocabs_to_dict(True),
        "probs": probs,
    }
    for k in kwargs.keys():
        result_dict.update({k: kwargs.get(k)})
    json.dump(result_dict, outfile)


def model_from_json(filepath: Path) -> NGramLanguageModel:
    with filepath.open("r") as file:
        obj = json.load(file)
    tokenizer = Tokenizer.load(obj["tokenizer"])
    model = NGramLanguageModel(tokenizer, obj["n"])
    probs = deserialize_probs(obj["probs"])
    model.probs = probs
    del obj["tokenizer"], obj["n"], obj["probs"]
    return model, obj
