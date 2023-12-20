import numpy as np


def perplexity(lm, tokenizer, lines, min_prob=10**-50.0):
    ppls = []
    for line in lines:
        log_ppl = 0
        tokenized = tokenizer(line)
        for i in range(len(tokenized)):
            log_ppl += np.log(
                max(min_prob, lm.get_token_prob(tokenized[i], tokenized[:i]))
            )
        ppls.append(np.exp(-log_ppl / len(tokenized)))

    return np.mean(ppls)
