from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import joblib

from util.jsonl_process import read_jsonl_basic


class BinarySentimentClassifier:
    """
    Бинарный классификатор тональности на основе логистической регрессии
    """

    def __init__(self, regularization='l2', C=1.0, positive_label=1, negative_label=0):
        """
        Args:
            regularization: 'l1' или 'l2' регуляризация
            C: параметр регуляризации (меньше = сильнее регуляризация)
            positive_label: метка для положительного класса
            negative_label: метка для отрицательного класса
        """
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )

        self.model = LogisticRegression(
            penalty=regularization,
            C=C,
            random_state=42,
            solver='liblinear' if regularization == 'l1' else 'lbfgs'
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
        Обучение модели
        """
        print("🎯 ОБУЧЕНИЕ БИНАРНОЙ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ...")

        # Подготовка данных
        X_train, y_train = self.prepare_data(train_data)

        # Проверяем, что у нас только 2 класса
        unique_labels = set(y_train)
        if len(unique_labels) != 2:
            print(f"⚠️  Предупреждение: обнаружено {len(unique_labels)} классов: {unique_labels}")
            print("   Убедитесь, что данные содержат только бинарные метки")

        # Векторизация текстов
        print("📊 Векторизация текстов...")
        X_train_vec = self.vectorizer.fit_transform(X_train)

        print(f"   Размерность признаков: {X_train_vec.shape}")
        print(f"   Классы: {unique_labels}")
        print(f"   Положительный класс: {self.positive_label}")
        print(f"   Отрицательный класс: {self.negative_label}")

        # Обучение модели
        print("🤖 Обучение модели...")
        self.model.fit(X_train_vec, y_train)
        self.is_trained = True

        # Оценка на тренировочных данных
        train_pred = self.model.predict(X_train_vec)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"✅ Точность на train: {train_accuracy:.3f}")

        # Оценка на валидации, если есть
        if val_data:
            val_accuracy = self.evaluate(val_data)
            print(f"✅ Точность на val: {val_accuracy:.3f}")

        # Покажем важные признаки
        self._show_important_features(top_n=10)

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

        # Для бинарной классификации у нас только 2 вероятности
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

        print("\n📈 МАТРИЦА ОШИБОК:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"               Предсказано {self.negative_label}  Предсказано {self.positive_label}")
        print(f"Реально {self.negative_label}:     {cm[0][0]:^10}        {cm[0][1]:^10}")
        print(f"Реально {self.positive_label}:     {cm[1][0]:^10}        {cm[1][1]:^10}")

        return accuracy

    def _show_important_features(self, top_n=10):
        """
        Показывает самые важные признаки для каждого класса
        """
        if not hasattr(self.model, 'coef_'):
            return

        feature_names = self.vectorizer.get_feature_names_out()

        print(f"\n🔍 ТОП-{top_n} ВАЖНЫХ ПРИЗНАКОВ:")

        # Для бинарной классификации у нас только один вектор коэффициентов
        coef = self.model.coef_[0]

        # Положительные признаки (указывают на положительный класс)
        print(f"\n   ПОЛОЖИТЕЛЬНЫЕ (указывают на класс {self.positive_label}):")
        pos_indices = np.argsort(coef)[-top_n:][::-1]
        for idx in pos_indices:
            print(f"      {feature_names[idx]}: {coef[idx]:.3f}")

        # Отрицательные признаки (указывают на отрицательный класс)
        print(f"\n   ОТРИЦАТЕЛЬНЫЕ (указывают на класс {self.negative_label}):")
        neg_indices = np.argsort(coef)[:top_n]
        for idx in neg_indices:
            print(f"      {feature_names[idx]}: {coef[idx]:.3f}")

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
        print(f"💾 Модель сохранена: {filename}")

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
        print(f"📥 Модель загружена: {filename}")


# Сравнение разных регуляризаций
def compare_regularizations(train_data, val_data):
    """
    Сравнение L1 и L2 регуляризаций
    """
    print("🔬 СРАВНЕНИЕ L1 vs L2 РЕГУЛЯРИЗАЦИИ")
    print("=" * 50)

    # L2 регуляризация (по умолчанию)
    print("\n1. L2 РЕГУЛЯРИЗАЦИЯ (Ridge):")
    l2_model = BinarySentimentClassifier(regularization='l2', C=1.0)
    l2_model.train(train_data, val_data)

    # L1 регуляризация (Lasso)
    print("\n2. L1 РЕГУЛЯРИЗАЦИЯ (Lasso):")
    l1_model = BinarySentimentClassifier(regularization='l1', C=1.0)
    l1_model.train(train_data, val_data)

    # Сравним количество ненулевых признаков
    l2_nonzero = np.sum(l2_model.model.coef_ != 0)
    l1_nonzero = np.sum(l1_model.model.coef_ != 0)

    print(f"\n📊 СРАВНЕНИЕ:")
    print(f"   L2 - ненулевых признаков: {l2_nonzero}")
    print(f"   L1 - ненулевых признаков: {l1_nonzero}")
    print(f"   L1 отбирает {l1_nonzero / l2_nonzero * 100:.1f}% признаков от L2")

    return l2_model, l1_model

def main():
    """
    Пример использования бинарного классификатора
    """

    train_data = read_jsonl_basic('../../util/news_sentiment_train.jsonl')
    val_data = read_jsonl_basic('../../util/news_sentiment_val.jsonl')
    test_data = read_jsonl_basic('../../util/news_sentiment_test.jsonl')

    print(f"📊 Данные: {len(train_data)} train, {len(val_data)} val")

    # 2. Обучаем модель с L2 регуляризацией
    print("\n" + "=" * 50)
    classifier = BinarySentimentClassifier(regularization='l2', C=1.0)
    classifier.train(train_data, val_data)

    # 3. Сохраняем модель
    classifier.save_model("binary_sentiment_classifier.pkl")

    # 4. Сравниваем регуляризации
    print("\n" + "=" * 50)
    l2_model, l1_model = compare_regularizations(train_data, val_data)


if __name__ == "__main__":
    main()