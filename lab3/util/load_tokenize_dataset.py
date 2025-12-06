from preprocessing.text_preprocessing import preprocess
from util.jsonl_process import read_jsonl_basic

def load_tokenize_ds(size: int = 10, lmtzr: bool = False, rm_stop_words: bool = True):
    file_name1 = "../dataset/old_dataset.jsonl"
    file_name2 = "../dataset/new_dataset.jsonl"

    data1 = read_jsonl_basic(file_name1)
    data2 = read_jsonl_basic(file_name2)

    data = data1 + data2

    data = data[:size]

    texts = [item['text'] for item in data]

    tokenize_texts = []
    for text in texts:
        tokenize_texts.append(preprocess(text, lmtzr=lmtzr, rm_stop_words=rm_stop_words))

    return tokenize_texts