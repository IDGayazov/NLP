from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import joblib

from util.jsonl_process import read_jsonl_basic


class SVMSentimentClassifier:
    """
    Бинарный классификатор тональности на основе SVM с линейным ядром
    """

    def __init__(self, C=1.0, loss='squared_hinge', penalty='l2', dual=True,
                 positive_label=1, negative_label=0, calibrate_probabilities=True):
        """
        Args:
            C: параметр регуляризации (меньше = сильнее регуляризация)
            loss: функция потерь ('hinge' или 'squared_hinge')
            penalty: тип регуляризации ('l1' или 'l2')
            dual: решать двойственную задачу (обычно True для kernel='linear')
            positive_label: метка для положительного класса
            negative_label: метка для отрицательного класса
            calibrate_probabilities: калибровать вероятности (рекомендуется для SVM)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),  # униграммы + биграммы
            stop_words=None  # SVM часто хорошо работает со стоп-словами
        )

        # Базовый SVM классификатор
        base_svm = LinearSVC(
            C=C,
            loss=loss,
            penalty=penalty,
            dual=dual,
            random_state=42,
            max_iter=1000  # Увеличиваем количество итераций для сходимости
        )

        # Калибровка вероятностей для SVM
        if calibrate_probabilities:
            self.model = CalibratedClassifierCV(base_svm, cv=3, method='sigmoid')
        else:
            self.model = base_svm

        self.positive_label = positive_label
        self.negative_label = negative_label
        self.calibrate_probabilities = calibrate_probabilities
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
        Обучение модели SVM
        """
        print("🎯 ОБУЧЕНИЕ SVM С ЛИНЕЙНЫМ ЯДРОМ...")

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
        print(f"   Калибровка вероятностей: {self.calibrate_probabilities}")

        # Обучение модели
        print("🤖 Обучение SVM...")
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
        self._show_important_features(top_n=15)

        # Информация о поддержке векторов (только для базового SVM)
        self._show_svm_info(X_train_vec)

    def predict(self, texts):
        """
        Предсказание для списка текстов
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)
        predictions = self.model.predict(X_vec)

        # Для SVM с калибровкой вероятностей
        if self.calibrate_probabilities:
            probabilities = self.model.predict_proba(X_vec)
        else:
            # Если вероятности не калиброваны, используем decision function
            decision_scores = self.model.decision_function(X_vec)
            # Преобразуем в псевдо-вероятности
            probabilities = self._decision_to_probability(decision_scores)

        return predictions, probabilities

    def _decision_to_probability(self, decision_scores):
        """
        Преобразование decision function в вероятности (простой способ)
        """
        # Простая сигмоидальная трансформация
        probabilities = 1 / (1 + np.exp(-decision_scores))
        # Создаем матрицу вероятностей для двух классов
        prob_matrix = np.zeros((len(probabilities), 2))
        prob_matrix[:, 1] = probabilities  # Вероятность положительного класса
        prob_matrix[:, 0] = 1 - probabilities  # Вероятность отрицательного класса
        return prob_matrix

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

        # Decision function для получения уверенности
        X_vec = self.vectorizer.transform([text])
        if hasattr(self.model, 'decision_function'):
            decision_score = self.model.decision_function(X_vec)[0]
        else:
            # Для калиброванной модели используем разность вероятностей
            decision_score = pos_prob - neg_prob

        return {
            'prediction': pred,
            'sentiment': sentiment,
            'positive_prob': pos_prob,
            'negative_prob': neg_prob,
            'confidence': abs(decision_score),  # Абсолютное значение decision function
            'decision_score': decision_score
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

    def _show_important_features(self, top_n=15):
        """
        Показывает самые важные признаки для каждого класса
        """
        # Получаем коэффициенты из базового estimator
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_[0]
        elif hasattr(self.model, 'estimators_') and hasattr(self.model.estimators_[0], 'coef_'):
            # Для CalibratedClassifierCV берем первый estimator
            coef = self.model.estimators_[0].coef_[0]
        else:
            print("❌ Не удалось получить коэффициенты модели")
            return

        feature_names = self.vectorizer.get_feature_names_out()

        print(f"\n🔍 ТОП-{top_n} ВАЖНЫХ ПРИЗНАКОВ SVM:")

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

    def _show_svm_info(self, X_train_vec):
        """
        Показывает информацию о векторах поддержки (только для LinearSVC)
        """
        try:
            if hasattr(self.model, 'estimators_'):
                # Для калиброванной модели
                base_estimator = self.model.estimators_[0]
            else:
                base_estimator = self.model

            if hasattr(base_estimator, 'support_'):
                n_support_vectors = len(base_estimator.support_)
                print(f"\n📊 ИНФОРМАЦИЯ О SVM:")
                print(f"   Количество векторов поддержки: {n_support_vectors}")
                print(f"   Процент от обучающей выборки: {n_support_vectors / len(X_train_vec) * 100:.1f}%")

        except Exception as e:
            print(f"   Информация о векторах поддержки недоступна: {e}")

    def get_decision_boundary_info(self, text):
        """
        Получить информацию о расстоянии до разделяющей гиперплоскости
        """
        X_vec = self.vectorizer.transform([text])

        if hasattr(self.model, 'decision_function'):
            decision_score = self.model.decision_function(X_vec)[0]
        else:
            # Для калиброванной модели
            decision_score = self.model.predict_proba(X_vec)[0][1] - 0.5

        distance_from_boundary = abs(decision_score)
        side = "positive" if decision_score > 0 else "negative"

        return {
            'decision_score': decision_score,
            'distance_from_boundary': distance_from_boundary,
            'side': side,
            'confidence': min(distance_from_boundary * 2, 1.0)  # Нормализованная уверенность
        }

    def save_model(self, filename):
        """
        Сохранение модели
        """
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'positive_label': self.positive_label,
            'negative_label': self.negative_label,
            'calibrate_probabilities': self.calibrate_probabilities
        }, filename)
        print(f"💾 Модель SVM сохранена: {filename}")

    def load_model(self, filename):
        """
        Загрузка модели
        """
        loaded = joblib.load(filename)
        self.model = loaded['model']
        self.vectorizer = loaded['vectorizer']
        self.positive_label = loaded.get('positive_label', 1)
        self.negative_label = loaded.get('negative_label', 0)
        self.calibrate_probabilities = loaded.get('calibrate_probabilities', True)
        self.is_trained = True
        print(f"📥 Модель SVM загружена: {filename}")


# Сравнение разных параметров SVM
def compare_svm_parameters(train_data, val_data):
    """
    Сравнение разных параметров SVM
    """
    print("🔬 СРАВНЕНИЕ ПАРАМЕТРОВ SVM")
    print("=" * 50)

    models = {}

    # 1. SVM с разными значениями C
    for C_value in [0.1, 1.0, 10.0]:
        print(f"\n1. SVM с C={C_value}:")
        model = SVMSentimentClassifier(C=C_value, calibrate_probabilities=False)
        model.train(train_data, val_data)
        models[f'SVM_C_{C_value}'] = model

    # 2. SVM с разными функциями потерь
    print(f"\n2. SVM с hinge loss:")
    model_hinge = SVMSentimentClassifier(loss='hinge', C=1.0, calibrate_probabilities=False)
    model_hinge.train(train_data, val_data)
    models['SVM_hinge'] = model_hinge

    return models


# Пример использования
def main():
    """
    Пример использования SVM классификатора
    """

    train_data = read_jsonl_basic('../../util/news_sentiment_train.jsonl')
    val_data = read_jsonl_basic('../../util/news_sentiment_val.jsonl')
    test_data = read_jsonl_basic('../../util/news_sentiment_test.jsonl')

    print(f"📊 Данные: {len(train_data)} train, {len(val_data)} val")

    # Обучаем модель SVM
    print("\n" + "=" * 50)
    svm_classifier = SVMSentimentClassifier(C=1.0, calibrate_probabilities=False)
    svm_classifier.train(train_data, val_data)

    # Сохраняем модель
    svm_classifier.save_model("svm_sentiment_classifier.pkl")

    # Сравниваем разные параметры
    print("\n" + "=" * 50)
    models = compare_svm_parameters(train_data[:100], val_data[:20])

if __name__ == "__main__":
    main()