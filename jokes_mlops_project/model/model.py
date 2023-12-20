from collections import Counter, defaultdict

import numpy as np


class NGramLanguageModel:
    def __init__(self, tokenizer, n=3):
        self.tokenizer = tokenizer
        self.n = n
        self.UNK = "[UNK]"
        self.EOS = "[EOS]"

    def __count_ngrams(self, corpus):
        counts = defaultdict(Counter)
        for text in corpus:
            tokenized = [self.UNK] * (self.n - 1) + self.tokenizer(text) + [self.EOS]
            for i in range(self.n - 1, len(tokenized)):
                counts[tuple(tokenized[i - self.n + 1 : i])][tokenized[i]] += 1
        return counts

    def fit(self, corpus):
        self.tokenizer.fit(corpus)
        counts = self.__count_ngrams(corpus)

        self.probs = defaultdict(Counter)

        for prefix, token_count in counts.items():
            token_sum = sum(token_count.values())
            for token, count in token_count.items():
                self.probs[prefix][token] = count / token_sum
        return self

    def process_prefix(self, prefix):
        if self.n == 1:
            prefix = []
        else:
            prefix = prefix[-(self.n - 1) :]
            prefix = [self.UNK] * (self.n - 1 - len(prefix)) + prefix

        return prefix

    def get_tokens_and_probs(self, prefix):
        prefix = self.process_prefix(prefix)
        possible_tokens = self.probs[tuple(prefix)]
        tokens = list(possible_tokens.keys())
        probs = list(possible_tokens.values())
        return tokens, probs

    def get_token_prob(self, token, prefix):
        prefix = self.process_prefix(prefix)
        prob = self.probs[tuple(prefix)].get(token, 0)
        return prob

    def get_next_token(self, prefix):
        tokens, probs = self.get_tokens_and_probs(prefix)
        next_token = np.random.choice(tokens, p=probs)
        return next_token

    def generate(self, prefix):
        prefix = self.tokenizer(prefix)
        for _ in range(100):
            prefix += [self.get_next_token(prefix)]
            if prefix[-1] == self.EOS or len(self.get_tokens_and_probs(prefix)[0]) == 0:
                break
        return " ".join(prefix)
