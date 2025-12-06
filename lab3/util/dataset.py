import json

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from util.decribe import get_dataset
from sklearn.model_selection import train_test_split
from collections import Counter

import pandas as pd

def make_dataset_for_classification():
    data = get_dataset()

    dataset = []
    for item in data:
        dataset.append({
            'title': item['title'],
            'text': item['text'],
            'sentiment': get_binary_class_from_category(item['rubric']),
            'category': item['rubric']
        })

    return dataset

def get_binary_class_from_category(category: str):
    if category in ('Югалту', 'Хәвеф-хәтәр', 'Махсус хәрби операция'):
        return 0
    else:
        return 1

def simple_train_val_test_split(data, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Простое разделение на train/validation/test с сохранением распределения меток

    Args:
        data: список словарей с данными
        train_size: доля тренировочных данных (0.7 = 70%)
        val_size: доля валидационных данных (0.15 = 15%)
        test_size: доля тестовых данных (0.15 = 15%)
        random_state: для воспроизводимости

    Returns:
        словарь с train, val, test наборами
    """

    # Проверяем, что сумма долей = 1
    assert abs(train_size + val_size + test_size - 1.0) < 0.001, "Сумма долей должна быть 1.0"

    # Извлекаем метки для стратификации
    labels = [item['sentiment'] for item in data]

    print("📊 Распределение меток в исходных данных:")
    label_counts = Counter(labels)
    for label, count in label_counts.items():
        print(f"   {label}: {count} ({count / len(data) * 100:.1f}%)")

    # 1. Сначала разделяем на train и temp (val + test)
    train_data, temp_data = train_test_split(
        data,
        test_size=val_size + test_size,
        stratify=labels,
        random_state=random_state
    )

    val_ratio = val_size / (val_size + test_size)

    val_data, test_data = train_test_split(
        temp_data,
        test_size=1 - val_ratio,
        stratify=[item['sentiment'] for item in temp_data],
        random_state=random_state
    )

    # Проверяем распределение в результатах
    print("\n✅ Проверка распределения после разделения:")
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        split_labels = [item['sentiment'] for item in split_data]
        label_dist = Counter(split_labels)
        print(f"\n   {split_name} ({len(split_data)} samples):")
        for label in sorted(label_dist.keys()):
            count = label_dist[label]
            pct = count / len(split_data) * 100
            original_pct = label_counts[label] / len(data) * 100
            deviation = abs(pct - original_pct)
            print(f"      {label}: {count} ({pct:.1f}%) - отклонение: {deviation:.1f}%")

    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

def train_val_test_split(data, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Простое разделение на train/validation/test с сохранением распределения меток

    Args:
        data: список словарей с данными
        train_size: доля тренировочных данных (0.7 = 70%)
        val_size: доля валидационных данных (0.15 = 15%)
        test_size: доля тестовых данных (0.15 = 15%)
        random_state: для воспроизводимости

    Returns:
        словарь с train, val, test наборами
    """

    # Проверяем, что сумма долей = 1
    assert abs(train_size + val_size + test_size - 1.0) < 0.001, "Сумма долей должна быть 1.0"

    new_data = []
    for item in data:
        if item['category'] != 'Югалту':
            new_data.append(item)

    data = new_data

    # Извлекаем метки для стратификации
    labels = [item['category'] for item in data]

    print("📊 Распределение меток в исходных данных:")
    label_counts = Counter(labels)
    for label, count in label_counts.items():
        print(f"   {label}: {count} ({count / len(data) * 100:.1f}%)")

    # 1. Сначала разделяем на train и temp (val + test)
    train_data, temp_data = train_test_split(
        data,
        test_size=val_size + test_size,
        stratify=labels,
        random_state=random_state
    )

    val_ratio = val_size / (val_size + test_size)

    val_data, test_data = train_test_split(
        temp_data,
        test_size=1 - val_ratio,
        stratify=[item['category'] for item in temp_data],
        random_state=random_state
    )

    # Проверяем распределение в результатах
    print("\n✅ Проверка распределения после разделения:")
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        split_labels = [item['category'] for item in split_data]
        label_dist = Counter(split_labels)
        print(f"\n   {split_name} ({len(split_data)} samples):")
        for label in sorted(label_dist.keys()):
            count = label_dist[label]
            pct = count / len(split_data) * 100
            original_pct = label_counts[label] / len(data) * 100
            deviation = abs(pct - original_pct)
            print(f"      {label}: {count} ({pct:.1f}%) - отклонение: {deviation:.1f}%")

    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }



def save_splits(splits, filename_prefix):
    """
    Сохраняем разделенные данные в файлы
    """
    for split_name, data in splits.items():
        filename = f"{filename_prefix}_{split_name}.jsonl"
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"💾 {split_name}: {len(data)} записей -> {filename}")


def create_multi_label_dataset(data, strategy='similarity', top_k=2):
    """
    Создание многометочного датасета из однолабельного

    Args:
        data: исходные данные
        strategy: 'similarity' (по семантической близости)
                 'cooccurrence' (по совместному появлению)
                 'hierarchical' (иерархическое разбиение)
        top_k: количество дополнительных меток
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    all_categories = sorted(list(set([item['category'] for item in data])))
    mlb = MultiLabelBinarizer(classes=all_categories)

    # Подсчитываем частоту категорий
    from collections import Counter
    cat_counter = Counter([item['category'] for item in data])
    print("📊 Распределение категорий:")
    for cat, count in cat_counter.most_common():
        print(f"  {cat}: {count} ({count / len(data) * 100:.1f}%)")

    if strategy == 'similarity':
        # Семантическая близость текстов
        print("🔍 Вычисление семантической близости...")
        texts = [item['text'] for item in data]

        # Используем SentenceTransformer для эмбеддингов
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        embeddings = model.encode(texts, show_progress_bar=True)

        # Матрица схожести
        similarity_matrix = cosine_similarity(embeddings)

        converted_data = []
        for i, item in enumerate(data):
            # Находим наиболее похожие тексты
            similarities = similarity_matrix[i]
            similar_indices = np.argsort(similarities)[::-1][1:top_k + 1]  # исключаем себя

            # Собираем метки из похожих текстов
            main_label = item['category']
            similar_labels = [data[idx]['category'] for idx in similar_indices]

            # Объединяем метки (уникальные)
            all_labels = list(set([main_label] + similar_labels))

            # Создаем бинарный вектор
            binary_vector = mlb.fit_transform([all_labels])[0]

            converted_data.append({
                'text': item['text'],
                'labels': all_labels,
                'binary_labels': binary_vector.tolist(),
                'main_category': main_label,
                'similar_categories': similar_labels
            })

    elif strategy == 'cooccurrence':
        # Создаем псевдо-совместное появление на основе категорий
        print("🔍 Создание псевдо-совместных появлений...")

        # Создаем матрицу совместного появления (псевдо)
        # В реальности нужно анализировать настоящие данные

        converted_data = []
        for item in data:
            main_label = item['category']

            # Добавляем наиболее частые категории
            most_common_cats = [cat for cat, _ in cat_counter.most_common(top_k + 3)
                                if cat != main_label][:top_k]

            all_labels = [main_label] + most_common_cats[:top_k]
            binary_vector = mlb.fit_transform([all_labels])[0]

            converted_data.append({
                'text': item['text'],
                'labels': all_labels,
                'binary_labels': binary_vector.tolist(),
                'main_category': main_label
            })

    return converted_data, all_categories, mlb


def convert_to_multi_label_format(data, label_type='binary'):
    """
    Преобразует датасет с одной меткой в формат для многометочной классификации

    Args:
        data: список словарей {'text': text, 'category': category}
        label_type: 'binary' (0/1) или 'hierarchical' (иерархическая)
    """
    all_categories = sorted(list(set([item['category'] for item in data])))
    num_classes = len(all_categories)

    print(f"📊 Найдено {num_classes} уникальных категорий:")
    for i, cat in enumerate(all_categories):
        print(f"  {i:3d}. {cat}")

    # Создаем энкодер
    mlb = MultiLabelBinarizer(classes=all_categories)

    converted_data = []
    for item in data:
        # Каждая запись получает только одну метку
        # Для multi-label преобразуем в список из одной метки
        labels = [item['category']]

        # Создаем бинарный вектор
        binary_vector = mlb.fit_transform([labels])[0]

        converted_item = {
            'text': item['text'],
            'labels': labels,  # оригинальные метки
            'binary_labels': binary_vector.tolist(),  # бинарный вектор
            'category': item['category']  # сохраняем для совместимости
        }
        converted_data.append(converted_item)

    return converted_data, all_categories, mlb

def main_binary():
    labeled_news = make_dataset_for_classification()

    splits = simple_train_val_test_split(
        labeled_news,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=42
    )

    save_splits(splits, "news_sentiment")

    print(f"\n🎯 ИТОГ:")
    print(f"   Train: {len(splits['train'])} записей (70%)")
    print(f"   Val:   {len(splits['val'])} записей (15%)")
    print(f"   Test:  {len(splits['test'])} записей (15%)")
    print(f"   Всего: {len(splits['train']) + len(splits['val']) + len(splits['test'])} записей")

def main_category():
    labeled_news = make_dataset_for_classification()

    splits = train_val_test_split(
        labeled_news,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=42
    )

    save_splits(splits, "news_category")

    print(f"\n🎯 ИТОГ:")
    print(f"   Train: {len(splits['train'])} записей (70%)")
    print(f"   Val:   {len(splits['val'])} записей (15%)")
    print(f"   Test:  {len(splits['test'])} записей (15%)")
    print(f"   Всего: {len(splits['train']) + len(splits['val']) + len(splits['test'])} записей")


def split_multi_label_data(conv_data, test_size=0.2, val_size=0.1, random_state=42):
    """
    Стратифицированное разделение многометочных данных

    Важно: нужно сохранять распределение меток во всех выборках!
    """

    # Извлекаем тексты и бинарные метки
    texts = [item['text'] for item in conv_data]
    binary_labels = np.array([item['binary_labels'] for item in conv_data])

    print(f"📊 Всего данных: {len(texts)}")
    print(f"📊 Размерность меток: {binary_labels.shape}")

    # 1. Сначала отделяем тестовую выборку (20%)
    # Используем итеративную стратификацию для многометочных данных
    from sklearn.model_selection import train_test_split

    # Для многометочных данных используем train_test_split несколько раз
    # или специальные методы

    # Временное решение: разделяем случайно, но с сохранением распределения меток
    # Для production лучше использовать IterativeStratification

    # Разделение: train_temp (80%) и test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, binary_labels,
        test_size=test_size,
        random_state=random_state,
        # Для многометочных простой random часто работает
        stratify=None  # к сожалению, стандартный stratify не работает для multi-label
    )

    # 2. Затем разделяем train_temp на train (70%) и val (10%)
    # Пересчитываем размер валидации относительно исходных данных
    val_relative_size = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=None
    )

    print(f"\n✅ Разделение завершено:")
    print(f"   Train: {len(X_train)} ({len(X_train) / len(texts) * 100:.1f}%)")
    print(f"   Val:   {len(X_val)} ({len(X_val) / len(texts) * 100:.1f}%)")
    print(f"   Test:  {len(X_test)} ({len(X_test) / len(texts) * 100:.1f}%)")

    # Проверяем распределение меток
    print(f"\n📊 Распределение меток (среднее количество на текст):")
    print(f"   Train: {y_train.sum(axis=1).mean():.2f}")
    print(f"   Val:   {y_val.sum(axis=1).mean():.2f}")
    print(f"   Test:  {y_test.sum(axis=1).mean():.2f}")

    # Проверяем покрытие классов
    print(f"\n📊 Покрытие классов (% текстов с данной меткой):")
    num_classes = binary_labels.shape[1]

    for i in range(min(5, num_classes)):  # покажем первые 5 классов
        train_coverage = (y_train[:, i].sum() / len(y_train)) * 100
        val_coverage = (y_val[:, i].sum() / len(y_val)) * 100
        test_coverage = (y_test[:, i].sum() / len(y_test)) * 100

        print(f"   Класс {i}: Train={train_coverage:.1f}%, "
              f"Val={val_coverage:.1f}%, Test={test_coverage:.1f}%")

    # Преобразуем обратно в удобный формат
    train_data = [
        {'text': text, 'binary_labels': labels.tolist()}
        for text, labels in zip(X_train, y_train)
    ]

    val_data = [
        {'text': text, 'binary_labels': labels.tolist()}
        for text, labels in zip(X_val, y_val)
    ]

    test_data = [
        {'text': text, 'binary_labels': labels.tolist()}
        for text, labels in zip(X_test, y_test)
    ]

    # splits = {
    #     'train_data': train_data,
    #     'test_data': test_data,
    #     'val_data': val_data
    # }
    # save_splits(splits, "news_multilabel")

    return train_data, val_data, test_data, binary_labels.shape[1]

if __name__ == "__main__":
    # main_binary()
    # main_category()
    data = make_dataset_for_classification()

    conv_data, _, _ = create_multi_label_dataset(data, strategy='cooccurrence')

    train_data, val_data, test_data, b_labels = split_multi_label_data(conv_data)

    print(b_labels)

    # multi_label_data, categories, mlb = convert_to_multi_label_format(data)
    #
    # print(f"\n📊 Пример преобразованной записи:")
    # print(f"Текст: {multi_label_data[0]['text'][:50]}...")
    # print(f"Исходная категория: {multi_label_data[0]['category']}")
    # print(f"Бинарный вектор ({len(categories)} классов): {multi_label_data[0]['binary_labels']}")
