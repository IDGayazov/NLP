# Описание данных
import string

from util.jsonl_process import read_jsonl_basic


def get_count(data):
    return len(data)

def get_unique_titles(data, accessor='title'):
    unique_names_count = len({item[accessor] for item in data})
    return unique_names_count

def get_unique_words_count(data):
    words = []

    for item in data:
        for word in item['text'].split(' '):
            words.append(word)

    words = set(words)
    return len(words)

def get_all_words(data):
    words = []

    for item in data:
        for word in item['text'].split(' '):
            words.append(word)

    return len(words)

def get_dataset():
    file_name = "../dataset/dataset.jsonl"
    data = read_jsonl_basic(file_name)
    return data

def get_texts():
    data = get_dataset()

    texts = []
    for item in data:
        texts.append(item['text'])
    return texts

def get_texts_app():
    file_name = "dataset/dataset.jsonl"
    data = read_jsonl_basic(file_name)

    texts = []
    for item in data:
        texts.append(item['text'])
    return texts

def get_labels():
    data = get_dataset()

    rubrics = []
    for item in data:
        rubrics.append(item['rubric'])
    return rubrics

def get_avg_word_len():
    data = get_dataset()

    words = []
    all_count = 0
    all_len = 0
    for item in data:
        for word in item['text'].split(' '):
            words.append(word)
            all_len += len(word)
            all_count += 1

    return all_len / all_count

def get_fraction_of_punctuation():
    data = get_dataset()

    all_count = 0
    punct_count = 0
    for item in data:
        for word in item['text']:
            all_count += 1
            for letter in word:
               if letter in string.punctuation:
                   punct_count += 1

    return punct_count / all_count

def get_fraction_of_upper():
    data = get_dataset()

    all_count = 0
    upper_count = 0
    for item in data:
        for word in item['text']:
            for letter in word:
                all_count += 1
                if letter.isupper():
                   upper_count += 1

    return upper_count / all_count

def get_fraction_of_digit():
    data = get_dataset()

    all_count = 0
    digit_count = 0
    for item in data:
        for word in item['text']:
            for letter in word:
                all_count += 1
                if letter.isdigit():
                   digit_count += 1

    return digit_count / all_count

def get_all_categories():
    data = get_dataset()

    cats = []
    for item in data:
        cats.append(item['rubric'])

    cats = set(cats)
    return cats

