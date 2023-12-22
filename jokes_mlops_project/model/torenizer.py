from bpe import Encoder


class Tokenizer:
    def __init__(self, dictionary_size, ngram_max=2, pct_bpe=0.95):
        self.bpe = Encoder(dictionary_size, ngram_max=ngram_max, pct_bpe=pct_bpe)

    def fit(self, corpus):
        self.bpe.fit(corpus)

    def tokenize(self, text):
        tokenized = self.bpe.tokenize(text)
        clear_tokenized = []
        first = False
        saw_eow = True
        for token in tokenized:
            if token == "__sow":
                saw_eow = False
                first = True
                continue
            elif token == "__eow":
                saw_eow = True
                continue
            else:
                if first or saw_eow:
                    clear_tokenized.append(token)
                    first = False
                else:
                    clear_tokenized.append("##" + token)
        return clear_tokenized

    def __call__(self, text):
        return self.tokenize(text)

    @classmethod
    def load(cls, tokenizer_dict):
        bpe = Encoder.from_dict(tokenizer_dict)
        tokenizer = cls(bpe.vocab_size, bpe.ngram_max, bpe.pct_bpe)
        tokenizer.bpe = bpe
        return tokenizer
