from tokenizers import ByteLevelBPETokenizer

from util.decribe import get_texts


class SimpleBPE:
    def __init__(self, vocab_size=1000):
        self.tokenizer = ByteLevelBPETokenizer()
        self.vocab_size = vocab_size

    def train(self, texts):
        """Обучение BPE на текстах"""
        self.tokenizer.train_from_iterator(
            texts,
            vocab_size=self.vocab_size,
            min_frequency=2
        )

    def tokenize(self, text):
        """Токенизация текста"""
        return self.tokenizer.encode(text).tokens


if __name__ == "__main__":
    texts = get_texts()
    bpe = SimpleBPE()
    bpe.train(texts)

    tokens = bpe.tokenize("Татарстан участковые Россиядәге иң яхшы участковыйлар унлыгына керде")
    print(tokens)