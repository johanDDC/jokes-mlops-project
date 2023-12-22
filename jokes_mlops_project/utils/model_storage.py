import json
from pathlib import Path

from model.model import NGramLanguageModel
from model.torenizer import Tokenizer


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


def model_to_json(model: NGramLanguageModel, outfile):
    probs = serialize_probs(model.probs)
    result_dict = {
        "n": model.n,
        "tokenizer": model.tokenizer.bpe.vocabs_to_dict(True),
        "probs": probs,
    }
    json.dump(result_dict, outfile)


def model_from_json(filepath: Path) -> NGramLanguageModel:
    with filepath.open("r") as file:
        obj = json.load(file)
    tokenizer = Tokenizer.load(obj["tokenizer"])
    model = NGramLanguageModel(tokenizer, obj["n"])
    probs = deserialize_probs(obj["probs"])
    model.probs = probs
    return model
