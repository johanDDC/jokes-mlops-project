from time import perf_counter

import numpy as np


primitives = {float, int, str}


def perplexity(language_model, lines, ids, min_prob=10**-50.0):
    ppls = []
    tokenizer = language_model.tokenizer
    for idx in ids:
        line = lines[idx]
        log_ppl = 0
        tokenized = tokenizer(line)
        for i in range(len(tokenized)):
            log_ppl += np.log(
                max(min_prob, language_model.get_token_prob(tokenized[i], tokenized[:i]))
            )
        ppls.append(np.exp(-log_ppl / len(tokenized)))

    return np.mean(ppls)


class Timer:
    def __enter__(self):
        self.start = perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time = perf_counter() - self.start
