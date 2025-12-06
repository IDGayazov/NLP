from tpot import TPOTClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import joblib
import time
import signal


def read_jsonl_basic(filepath):
    """
    Чтение JSONL файла
    """
    import json
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data


class TimeoutTPOT:
    """
    TPOT с таймаутом чтобы не зависал
    """

    def __init__(self, time_minutes=1):
        self.vectorizer = TfidfVectorizer(
            max_features=500,  # Еще меньше фичей
            min_df=1,
            max_df=1.0
        )
        self.time_minutes = time_minutes
        self.is_trained = False
        print(f"⏰ TPOT с таймаутом: {time_minutes} мин")

    def train_with_timeout(self, train_data):
        """Обучение с таймаутом"""
        texts = [item['text'] for item in train_data]
        labels = [item['sentiment'] for item in train_data]

        X = self.vectorizer.fit_transform(texts).toarray()
        y = np.array(labels)

        print(f"📊 Данные: {X.shape}")

        # Функция для обработки таймаута
        def timeout_handler(signum, frame):
            raise TimeoutError("TPOT превысил время обучения!")

        # Устанавливаем таймаут
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.time_minutes * 60 + 10)  # +10 секунд буфер

        try:
            print("🔄 Запускаем TPOT...")
            self.tpot = TPOTClassifier(
                max_time_mins=self.time_minutes,
                random_state=42,
                population_size=5,  # Очень маленькая популяция
                generations=2  # Всего 2 поколения
            )
            self.tpot.fit(X, y)
            self.is_trained = True
            signal.alarm(0)  # Отключаем таймаут
            print("✅ TPOT успешно обучился!")

        except TimeoutError:
            print("⏰ TPOT превысил время! Используем fallback...")
            self._fallback_training(X, y)
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            self._fallback_training(X, y)

    def _fallback_training(self, X, y):
        """Резервное обучение если TPOT не работает"""
        from sklearn.linear_model import LogisticRegression

        print("🔄 Используем LogisticRegression как fallback...")
        self.fallback_model = LogisticRegression(random_state=42)
        self.fallback_model.fit(X, y)
        self.is_trained = True
        self.use_fallback = True

        score = self.fallback_model.score(X, y)
        print(f"✅ Fallback модель готова. Точность: {score:.3f}")

    def evaluate(self, test_data):
        """Оценка модели"""
        texts = [item['text'] for item in test_data]
        labels = [item['sentiment'] for item in test_data]

        X = self.vectorizer.transform(texts).toarray()
        y = np.array(labels)

        if hasattr(self, 'use_fallback') and self.use_fallback:
            predictions = self.fallback_model.predict(X)
            model_type = "LogisticRegression (fallback)"
        else:
            predictions = self.tpot.predict(X)
            model_type = "TPOT"

        accuracy = accuracy_score(y, predictions)

        print(f"📈 Результаты ({model_type}):")
        print(f"   Точность: {accuracy:.3f}")
        print(f"   Отчет:")
        print(classification_report(y, predictions))

        return accuracy


# Альтернатива: используем только LogisticRegression
class SimpleClassifier:
    """
    Простой классификатор без TPOT
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=1,
            max_df=1.0
        )
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(random_state=42)
        print("🤖 Простой LogisticRegression классификатор")

    def train(self, train_data):
        """Быстрое обучение"""
        texts = [item['text'] for item in train_data]
        labels = [item['sentiment'] for item in train_data]

        print(f"📚 Обучаем на {len(texts)} примерах...")

        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)

        self.model.fit(X, y)

        train_score = self.model.score(X, y)
        print(f"✅ Обучение завершено. Точность: {train_score:.3f}")

    def evaluate(self, test_data):
        """Оценка"""
        texts = [item['text'] for item in test_data]
        labels = [item['sentiment'] for item in test_data]

        X = self.vectorizer.transform(texts)
        y = np.array(labels)

        predictions = self.model.predict(X)
        accuracy = accuracy_score(y, predictions)

        print(f"📊 Точность на тесте: {accuracy:.3f}")
        print("\n📈 Детальный отчет:")
        print(classification_report(y, predictions))

        return accuracy

    def predict_text(self, text):
        """Предсказание"""
        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]

        return {
            'prediction': prediction,
            'confidence': f"{max(probability):.3f}",
            'probabilities': {
                f'class_{i}': f"{prob:.3f}" for i, prob in enumerate(probability)
            }
        }


# Основная функция
def main():
    # Загрузка данных
    train_data = read_jsonl_basic('../../util/news_sentiment_train.jsonl')
    test_data = read_jsonl_basic('../../util/news_sentiment_test.jsonl')

    print("=" * 50)
    print("🚀 КЛАССИФИКАТОРЫ ДЛЯ ТЕКСТА")
    print("=" * 50)
    print(f"📁 Train: {len(train_data)} примеров")
    print(f"📁 Test: {len(test_data)} примеров")

    # Вариант 1: Простой и быстрый классификатор
    print("\n🎯 ВАРИАНТ 1: ПРОСТОЙ КЛАССИФИКАТОР")
    simple_model = SimpleClassifier()
    simple_model.train(train_data)
    simple_accuracy = simple_model.evaluate(test_data)

    # Вариант 2: TPOT с таймаутом (опционально)
    use_tpot = input("\n🤔 Попробовать TPOT? (y/n): ").lower().strip() == 'y'

    if use_tpot:
        print("\n🎯 ВАРИАНТ 2: TPOT С ТАЙМАУТОМ")
        try:
            tpot_model = TimeoutTPOT(time_minutes=1)  # Всего 1 минута
            tpot_model.train_with_timeout(train_data)
            tpot_accuracy = tpot_model.evaluate(test_data)
        except Exception as e:
            print(f"❌ TPOT не сработал: {e}")
            tpot_accuracy = 0
    else:
        tpot_accuracy = 0

    # Сравнение результатов
    print("\n📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ:")
    print(f"   SimpleClassifier: {simple_accuracy:.3f}")
    if tpot_accuracy > 0:
        print(f"   TPOT: {tpot_accuracy:.3f}")

    # Сохраняем простую модель
    joblib.dump(simple_model, "models/simple_classifier.pkl")
    print("💾 Простая модель сохранена как 'simple_classifier.pkl'")


if __name__ == "__main__":
    # Запускаем основную версию
    main()