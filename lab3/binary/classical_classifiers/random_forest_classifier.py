from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import joblib

from util.jsonl_process import read_jsonl_basic


class RandomForestSentimentClassifier:
    """
    Бинарный классификатор тональности на основе случайного леса
    """

    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt',
                 positive_label=1, negative_label=0, random_state=42):
        """
        Args:
            n_estimators: количество деревьев в лесу
            max_depth: максимальная глубина деревьев
            max_features: количество признаков для рассмотрения в каждом разбиении
            positive_label: метка для положительного класса
            negative_label: метка для отрицательного класса
            random_state: для воспроизводимости результатов
        """
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,  # Используем все ядра процессора
            bootstrap=True,
            oob_score=True  # Out-of-bag score для оценки качества
        )

        self.positive_label = positive_label
        self.negative_label = negative_label
        self.is_trained = False

    def prepare_data(self, data):
        """
        Подготовка данных: извлекаем тексты и метки
        """
        texts = [item['text'] for item in data]
        labels = [item['sentiment'] for item in data]
        return texts, labels

    def train(self, train_data, val_data=None):
        """
        Обучение модели случайного леса
        """
        print("🎯 ОБУЧЕНИЕ СЛУЧАЙНОГО ЛЕСА...")

        # Подготовка данных
        X_train, y_train = self.prepare_data(train_data)

        # Проверяем, что у нас только 2 класса
        unique_labels = set(y_train)
        if len(unique_labels) != 2:
            print(f"⚠️  Предупреждение: обнаружено {len(unique_labels)} классов: {unique_labels}")

        # Векторизация текстов
        print("📊 Векторизация текстов...")
        X_train_vec = self.vectorizer.fit_transform(X_train)

        print(f"   Размерность признаков: {X_train_vec.shape}")
        print(f"   Классы: {unique_labels}")
        print(f"   Количество деревьев: {self.model.n_estimators}")
        print(f"   Максимальная глубина: {self.model.max_depth}")

        # Обучение модели
        print("🤖 Обучение случайного леса...")
        self.model.fit(X_train_vec, y_train)
        self.is_trained = True

        # Оценка на тренировочных данных
        train_pred = self.model.predict(X_train_vec)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"✅ Точность на train: {train_accuracy:.3f}")

        # Out-of-bag score
        if hasattr(self.model, 'oob_score_'):
            print(f"✅ Out-of-bag score: {self.model.oob_score_:.3f}")

        # Оценка на валидации, если есть
        if val_data:
            val_accuracy = self.evaluate(val_data)
            print(f"✅ Точность на val: {val_accuracy:.3f}")

        # Покажем важные признаки
        self._show_important_features(top_n=20)

        # Информация о модели
        self._show_model_info()

    def predict(self, texts):
        """
        Предсказание для списка текстов
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)
        predictions = self.model.predict(X_vec)
        probabilities = self.model.predict_proba(X_vec)

        return predictions, probabilities

    def predict_single(self, text):
        """
        Предсказание для одного текста с детальной информацией
        """
        predictions, probabilities = self.predict([text])
        pred = predictions[0]
        prob = probabilities[0]

        # Определяем вероятности для каждого класса
        if self.model.classes_[0] == self.positive_label:
            pos_prob = prob[0]
            neg_prob = prob[1]
        else:
            pos_prob = prob[1]
            neg_prob = prob[0]

        sentiment = "POSITIVE" if pred == self.positive_label else "NEGATIVE"

        return {
            'prediction': pred,
            'sentiment': sentiment,
            'positive_prob': pos_prob,
            'negative_prob': neg_prob,
            'confidence': max(pos_prob, neg_prob)
        }

    def evaluate(self, test_data):
        """
        Оценка модели на тестовых данных
        """
        X_test, y_test = self.prepare_data(test_data)
        X_test_vec = self.vectorizer.transform(X_test)

        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)

        print("\n📊 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        target_names = [f'NEGATIVE({self.negative_label})', f'POSITIVE({self.positive_label})']
        print(classification_report(y_test, y_pred, target_names=target_names))

        # Матрица ошибок
        print("\n📈 МАТРИЦА ОШИБОК:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"               Предсказано {self.negative_label}  Предсказано {self.positive_label}")
        print(f"Реально {self.negative_label}:     {cm[0][0]:^10}        {cm[0][1]:^10}")
        print(f"Реально {self.positive_label}:     {cm[1][0]:^10}        {cm[1][1]:^10}")

        return accuracy

    def _show_important_features(self, top_n=20):
        """
        Показывает самые важные признаки
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("❌ Не удалось получить важность признаков")
            return

        feature_names = self.vectorizer.get_feature_names_out()
        importances = self.model.feature_importances_

        print(f"\n🔍 ТОП-{top_n} ВАЖНЫХ ПРИЗНАКОВ (Random Forest):")

        # Сортируем признаки по важности
        indices = np.argsort(importances)[::-1]

        print(f"\n   САМЫЕ ВАЖНЫЕ ПРИЗНАКИ:")
        for i in range(min(top_n, len(indices))):
            idx = indices[i]
            print(f"      {feature_names[idx]}: {importances[idx]:.4f}")

        # Покажем распределение важности
        total_importance = np.sum(importances)
        top_n_importance = np.sum(importances[indices[:top_n]])
        print(f"\n   📊 Топ-{top_n} признаков объясняют {top_n_importance / total_importance * 100:.1f}% общей важности")

    def _show_model_info(self):
        """
        Показывает информацию о обученной модели
        """
        print(f"\n📊 ИНФОРМАЦИЯ О СЛУЧАЙНОМ ЛЕСЕ:")
        print(f"   Количество деревьев: {len(self.model.estimators_)}")
        print(f"   Глубина деревьев: {max([est.tree_.max_depth for est in self.model.estimators_])} (макс)")

        # Среднее количество листьев
        n_leaves = [est.tree_.n_leaves for est in self.model.estimators_]
        print(f"   Листья в дереве: {np.mean(n_leaves):.0f} (в среднем)")

        if hasattr(self.model, 'oob_score_'):
            print(f"   Out-of-bag score: {self.model.oob_score_:.3f}")

    def get_feature_importance_df(self, top_n=50):
        """
        Возвращает DataFrame с важностью признаков
        """
        if not hasattr(self.model, 'feature_importances_'):
            return None

        feature_names = self.vectorizer.get_feature_names_out()
        importances = self.model.feature_importances_

        # Сортируем по важности
        indices = np.argsort(importances)[::-1]

        importance_data = []
        for i in range(min(top_n, len(indices))):
            idx = indices[i]
            importance_data.append({
                'feature': feature_names[idx],
                'importance': importances[idx],
                'rank': i + 1
            })

        return importance_data

    def save_model(self, filename):
        """
        Сохранение модели
        """
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'positive_label': self.positive_label,
            'negative_label': self.negative_label
        }, filename)
        print(f"💾 Модель случайного леса сохранена: {filename}")

    def load_model(self, filename):
        """
        Загрузка модели
        """
        loaded = joblib.load(filename)
        self.model = loaded['model']
        self.vectorizer = loaded['vectorizer']
        self.positive_label = loaded.get('positive_label', 1)
        self.negative_label = loaded.get('negative_label', 0)
        self.is_trained = True
        print(f"📥 Модель случайного леса загружена: {filename}")


# Сравнение разных параметров случайного леса
def compare_rf_parameters(train_data, val_data):
    """
    Сравнение разных параметров случайного леса
    """
    print("🔬 СРАВНЕНИЕ ПАРАМЕТРОВ СЛУЧАЙНОГО ЛЕСА")
    print("=" * 50)

    models = {}

    # 1. Разное количество деревьев
    for n_trees in [50, 100, 200]:
        print(f"\n1. Random Forest с {n_trees} деревьями:")
        model = RandomForestSentimentClassifier(n_estimators=n_trees, max_depth=None)
        model.train(train_data, val_data)
        models[f'RF_{n_trees}trees'] = model

    # 2. Разная максимальная глубина
    print(f"\n2. Random Forest с ограничением глубины (max_depth=10):")
    model_shallow = RandomForestSentimentClassifier(n_estimators=100, max_depth=10)
    model_shallow.train(train_data, val_data)
    models['RF_depth10'] = model_shallow

    return models


# Анализ важности признаков
def analyze_feature_importance(model, top_n=30):
    """
    Детальный анализ важности признаков
    """
    importance_data = model.get_feature_importance_df(top_n=top_n)

    if importance_data:
        print(f"\n📈 ДЕТАЛЬНЫЙ АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ (Топ-{top_n}):")
        print("=" * 60)

        for i, item in enumerate(importance_data[:top_n]):
            print(f"{i + 1:2d}. {item['feature']:20s} : {item['importance']:.4f}")

        # Группируем по типам признаков
        positive_words = []
        negative_words = []

        for item in importance_data:
            feature = item['feature']
            # Простая эвристика для определения тональности признака
            if any(word in feature for word in ['хорош', 'отлич', 'прекрас', 'довол', 'рекоменд']):
                positive_words.append(item)
            elif any(word in feature for word in ['плох', 'ужас', 'разочар', 'недовол', 'проблем']):
                negative_words.append(item)

        print(f"\n🎯 ПОЛОЖИТЕЛЬНЫЕ ПРИЗНАКИ:")
        for item in positive_words[:10]:
            print(f"   {item['feature']}: {item['importance']:.4f}")

        print(f"\n🎯 ОТРИЦАТЕЛЬНЫЕ ПРИЗНАКИ:")
        for item in negative_words[:10]:
            print(f"   {item['feature']}: {item['importance']:.4f}")


# Пример использования
def main():
    """
    Пример использования случайного леса для классификации тональности
    """
    train_data = read_jsonl_basic('../../util/news_sentiment_train.jsonl')
    val_data = read_jsonl_basic('../../util/news_sentiment_val.jsonl')
    test_data = read_jsonl_basic('../../util/news_sentiment_test.jsonl')

    # Обучаем модель случайного леса
    print("\n" + "=" * 50)
    rf_classifier = RandomForestSentimentClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    rf_classifier.train(train_data, val_data)

    # Детальный анализ важности признаков
    analyze_feature_importance(rf_classifier, top_n=25)

    # Сохраняем модель
    rf_classifier.save_model("random_forest_sentiment_classifier.pkl")

    # Сравниваем разные параметры
    print("\n" + "=" * 50)
    models = compare_rf_parameters(train_data[:200], val_data[:40])  # Подмножество для скорости


# Простой способ быстро обучить модель
def quick_train_rf(train_file, val_file=None, n_estimators=100):
    """
    Быстрое обучение случайного леса из файлов
    """
    import json

    # Загружаем данные
    def load_jsonl(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file) if val_file else None

    # Проверяем, что данные бинарные
    unique_labels = set(item['sentiment'] for item in train_data)
    if len(unique_labels) != 2:
        print(f"⚠️  Ошибка: данные содержат {len(unique_labels)} классов, но требуется 2")
        return None

    # Обучаем модель случайного леса
    classifier = RandomForestSentimentClassifier(n_estimators=n_estimators)
    classifier.train(train_data, val_data)

    return classifier


if __name__ == "__main__":
    main()