from collections import Counter, defaultdict

import numpy as np


class NGramLanguageModel:
    def __init__(self, tokenizer, n_grams=3):
        self.tokenizer = tokenizer
        self.n_grams = n_grams
        self.UNK = "[UNK]"
        self.EOS = "[EOS]"

    def __count_ngrams(self, corpus):
        counts = defaultdict(Counter)
        for text in corpus:
            tokenized = (
                [self.UNK] * (self.n_grams - 1) + self.tokenizer(text) + [self.EOS]
            )
            for i in range(self.n_grams - 1, len(tokenized)):
                counts[tuple(tokenized[i - self.n_grams + 1 : i])][tokenized[i]] += 1
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
        if self.n_grams == 1:
            prefix = []
        else:
            prefix = prefix[-(self.n_grams - 1) :]
            prefix = [self.UNK] * (self.n_grams - 1 - len(prefix)) + prefix

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

    def __post_process(self, text):
        punkt = {",", ".", "?", "!", ":"}
        final = []
        for word in text:
            if word[0] == "#" or word in punkt:
                word = word.replace("#", "")
                if len(final) == 0:
                    final.append(word)
                else:
                    final[-1] = final[-1] + word
            else:
                if word != self.EOS:
                    final.append(word)
        return " ".join(final)

    def generate(self, prefix):
        prefix = self.tokenizer(prefix)
        for _ in range(100):
            prefix += [self.get_next_token(prefix)]
            if prefix[-1] == self.EOS or len(self.get_tokens_and_probs(prefix)[0]) == 0:
                break
        return self.__post_process(prefix)
