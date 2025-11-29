import re
import html
from typing import List

import emoji

from preprocessing.spacy_tokenizer import SpacyTokenizer
from preprocessing.tatar_lemmatizer import TatarLemmatizer
from preprocessing.tatar_stop_words_remover import TatarStopWordsRemover
from util.jsonl_process import read_jsonl_basic


def clean_text_html(text: str):
    """
    Очистка текста только с помощью регулярных выражений
    """
    if not text:
        return ""

    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\(\)\"]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def lowercase(text: str):
    return text.lower()

def emoji_to_text(text: str):
    """
    Замена эмодзи на текстовое описание с помощью библиотеки emoji
    """
    if not text:
        return text

    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.replace(':', ' ').replace('_', ' ')
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def tokenizer(text: str) -> List[str]:
    tkzr = SpacyTokenizer()
    return tkzr.tokenize_text(text)

def lemmatizer(word: str) -> str:
    lmtzr = TatarLemmatizer()
    return lmtzr.rule_based_lemmatize(word)

def remove_stop_words(tokens: List[str]) -> List[str]:
    remover = TatarStopWordsRemover()
    return remover.remove_tatar_stopwords(tokens)

def preprocess(text: str, lmtzr=False, rm_stop_words=True) -> List[str]:
    text = clean_text_html(text)
    text = lowercase(text)
    text = emoji_to_text(text)

    tokens = tokenizer(text)

    if lmtzr:
        tokens = [lemmatizer(token) for token in tokens]

    if rm_stop_words:
        return remove_stop_words(tokens)

    return tokens

if __name__ == "__main__":
    test1 = "<a>Hello, world!</a>"

    print("Before: " + test1)
    print("After: " + clean_text_html(test1))

    test2 = "UPPER TEXT"
    print("Before: " + test2)
    print("After: " + lowercase(test2))

    test3 = "Я люблю программировать! 😊 Python это круто! 🐍🔥"
    print("Before: " + test3)
    print("After: " + emoji_to_text(test3))

    file_name = "../dataset/old_dataset.jsonl"

    test4 = read_jsonl_basic(file_name)
    print(preprocess(test4[0]['text'], lmtzr=True))