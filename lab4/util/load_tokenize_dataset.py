from preprocessing.text_preprocessing import preprocess
from util.decribe import get_dataset

def load_tokenize_ds(size: int = 10, lmtzr: bool = False, rm_stop_words: bool = True):
    data = get_dataset()
    data = data[:size]
    texts = [item['text'] for item in data]

    tokenize_texts = []
    for text in texts:
        tokenize_texts.append(preprocess(text, lmtzr=lmtzr, rm_stop_words=rm_stop_words))

    return tokenize_texts