import json


def read_jsonl_basic(filename):
    """Чтение JSONL файла"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def convert_to_jsonl(data_list, filename):
    """
    Конвертирует список словарей в формат JSONL

    Args:
        data_list: список словарей с данными
        filename: имя файла для сохранения
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')