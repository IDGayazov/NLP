import json
import text_cleaner
from universal_preprocessor import quick_preprocess
from tokenization import TatarTokenizationExperiment


def read_jsonl_to_list(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:  # пропускаем пустые строки
                data.append(json.loads(line))
    return data


if __name__ == '__main__':
    data_list = read_jsonl_to_list('lenta_news_20251006_192459.jsonl')

    cleaner = text_cleaner.TextCleaner(language='tatar')

    cleaned_data = []

    for data in data_list:
        data['text'] = cleaner.clean_text(data['text'])
        cleaned_data.append(data)

    preprocessed_data = []

    for data in cleaned_data:
        data['text'] = quick_preprocess(data['text'])
        preprocessed_data.append(data)

    full_text = ""
    for data in preprocessed_data:
        full_text = full_text + " " + data['text']

    tknz = TatarTokenizationExperiment(full_text)
    tknz.run_experiment()

