from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import joblib

from util.jsonl_process import read_jsonl_basic


class BaggingSentimentClassifier:
    """
    Бинарный классификатор тональности на основе Bagging ансамбля
    """

    def __init__(self, base_estimator='logistic', n_estimators=10,
                 max_samples=1.0, max_features=1.0, bootstrap=True,
                 bootstrap_features=False, random_state=42,
                 positive_label=1, negative_label=0):
        """
        Args:
            base_estimator: 'logistic' или 'tree' - базовый алгоритм
            n_estimators: количество базовых классификаторов
            max_samples: доля/количество samples для каждого классификатора
            max_features: доля/количество features для каждого классификатора
            bootstrap: использовать ли bootstrap sampling
            bootstrap_features: использовать ли bootstrap для features
            random_state: для воспроизводимости
            positive_label: метка положительного класса
            negative_label: метка отрицательного класса
        """
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )

        # Выбор базового классификатора
        if base_estimator == 'logistic':
            base_est = LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                C=1.0
            )
        elif base_estimator == 'tree':
            base_est = DecisionTreeClassifier(
                random_state=random_state,
                max_depth=None
            )
        else:
            raise ValueError("base_estimator должен быть 'logistic' или 'tree'")

        self.model = BaggingClassifier(
            estimator=base_est,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            random_state=random_state,
            n_jobs=-1,  # Используем все ядра
            verbose=0
        )

        self.base_estimator = base_estimator
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
        Обучение Bagging модели
        """
        print(f"🎯 ОБУЧЕНИЕ BAGGING ({self.base_estimator.upper()})...")

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
        print(f"   Базовый алгоритм: {self.base_estimator}")
        print(f"   Количество моделей: {self.model.n_estimators}")
        print(f"   Max samples: {self.model.max_samples}")
        print(f"   Max features: {self.model.max_features}")
        print(f"   Bootstrap: {self.model.bootstrap}")
        print(f"   Bootstrap features: {self.model.bootstrap_features}")

        # Обучение модели
        print("🤖 Обучение Bagging ансамбля...")
        self.model.fit(X_train_vec, y_train)
        self.is_trained = True

        # Оценка на тренировочных данных
        train_pred = self.model.predict(X_train_vec)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"✅ Точность на train: {train_accuracy:.3f}")

        # Out-of-bag оценка (если bootstrap=True)
        if hasattr(self.model, 'oob_score_') and self.model.oob_score:
            print(f"✅ Out-of-bag score: {self.model.oob_score_:.3f}")

        # Оценка на валидации, если есть
        if val_data:
            val_accuracy = self.evaluate(val_data)
            print(f"✅ Точность на val: {val_accuracy:.3f}")

        # Анализ ансамбля
        self._analyze_ensemble()

        # Покажем важные признаки (если возможно)
        self._show_important_features(top_n=15)

    def predict(self, texts):
        """
        Предсказание для списка текстов
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)

        # Для BaggingClassifier predict_proba доступен
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

    def predict_with_consensus(self, texts):
        """
        Предсказание с информацией о консенсусе ансамбля
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)

        # Получаем предсказания всех базовых моделей
        all_predictions = []

        for i, estimator in enumerate(self.model.estimators_):
            try:
                # Если используются подмножества признаков, выбираем соответствующие фичи
                if hasattr(self.model, 'estimators_features_') and self.model.estimators_features_:
                    features_idx = self.model.estimators_features_[i]
                    X_subset = X_vec[:, features_idx]
                else:
                    X_subset = X_vec

                predictions = estimator.predict(X_subset)
                all_predictions.append(predictions)
            except Exception as e:
                print(f"⚠️  Ошибка в модели {i}: {e}")
                continue

        if not all_predictions:
            raise Exception("Ни одна модель не смогла сделать предсказание")

        all_predictions = np.array(all_predictions)

        # Основное предсказание
        final_predictions = self.model.predict(X_vec)
        probabilities = self.model.predict_proba(X_vec)

        results = []
        for i, text in enumerate(texts):
            # Считаем голоса
            votes = all_predictions[:, i]
            positive_votes = np.sum(votes == self.positive_label)
            negative_votes = np.sum(votes == self.negative_label)
            total_votes = len(votes)

            consensus_ratio = max(positive_votes, negative_votes) / total_votes

            results.append({
                'prediction': final_predictions[i],
                'probability': probabilities[i],
                'positive_votes': positive_votes,
                'negative_votes': negative_votes,
                'total_votes': total_votes,
                'consensus_ratio': consensus_ratio,
                'unanimous': consensus_ratio == 1.0
            })

        return results

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

    def _analyze_ensemble(self):
        """
        Анализ разнообразия и качества ансамбля
        """
        print(f"\n📊 АНАЛИЗ BAGGING АНСАМБЛЯ:")
        print(f"   Количество моделей: {len(self.model.estimators_)}")

        # Оценим разнообразие ансамбля
        if hasattr(self.model, 'estimators_features_'):
            unique_features_sets = len(set(
                tuple(features) for features in self.model.estimators_features_
            ))
            print(f"   Уникальных наборов признаков: {unique_features_sets}")

        # Out-of-bag score
        if hasattr(self.model, 'oob_score_'):
            print(f"   Out-of-bag score: {self.model.oob_score_:.3f}")

    def _show_important_features(self, top_n=15):
        """
        Показывает важные признаки (для логистической регрессии)
        """
        if self.base_estimator != 'logistic':
            print(f"\n⚠️  Важность признаков недоступна для базового {self.base_estimator}")
            return

        try:
            # Для логистической регрессии усредняем коэффициенты по всем моделям
            all_coefs = []
            for estimator in self.model.estimators_:
                if hasattr(estimator, 'coef_'):
                    all_coefs.append(estimator.coef_[0])

            if not all_coefs:
                print("❌ Не удалось получить коэффициенты моделей")
                return

            # Усредненные коэффициенты
            avg_coefs = np.mean(all_coefs, axis=0)
            feature_names = self.vectorizer.get_feature_names_out()

            print(f"\n🔍 ТОП-{top_n} ВАЖНЫХ ПРИЗНАКОВ (Bagging Logistic):")

            # Положительные признаки
            print(f"\n   ПОЛОЖИТЕЛЬНЫЕ (указывают на класс {self.positive_label}):")
            pos_indices = np.argsort(avg_coefs)[-top_n:][::-1]
            for idx in pos_indices:
                print(f"      {feature_names[idx]}: {avg_coefs[idx]:.3f}")

            # Отрицательные признаки
            print(f"\n   ОТРИЦАТЕЛЬНЫЕ (указывают на класс {self.negative_label}):")
            neg_indices = np.argsort(avg_coefs)[:top_n]
            for idx in neg_indices:
                print(f"      {feature_names[idx]}: {avg_coefs[idx]:.3f}")

        except Exception as e:
            print(f"❌ Ошибка при анализе важности признаков: {e}")

    def get_ensemble_diversity(self, data):
        """
        Оценивает разнообразие ансамбля
        """
        X, y = self.prepare_data(data)
        X_vec = self.vectorizer.transform(X)

        # Получаем предсказания всех моделей
        all_predictions = []

        for i, estimator in enumerate(self.model.estimators_):
            try:
                if hasattr(self.model, 'estimators_features_') and self.model.estimators_features_:
                    features_idx = self.model.estimators_features_[i]
                    X_subset = X_vec[:, features_idx]
                else:
                    X_subset = X_vec

                predictions = estimator.predict(X_subset)
                all_predictions.append(predictions)
            except Exception as e:
                print(f"⚠️  Ошибка в модели {i} при оценке разнообразия: {e}")
                continue

        if not all_predictions:
            return {'diversity_score': 0, 'average_disagreement': 0, 'n_models': 0}

        all_predictions = np.array(all_predictions)
        n_models = len(all_predictions)

        # Считаем попарные различия
        disagreements = 0
        total_pairs = 0

        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreements += np.sum(all_predictions[i] != all_predictions[j])
                total_pairs += len(y)

        diversity_score = disagreements / total_pairs if total_pairs > 0 else 0

        return {
            'diversity_score': diversity_score,
            'average_disagreement': disagreements / (n_models * (n_models - 1) / 2) if n_models > 1 else 0,
            'n_models': n_models
        }

    def save_model(self, filename):
        """
        Сохранение модели
        """
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'base_estimator': self.base_estimator,
            'positive_label': self.positive_label,
            'negative_label': self.negative_label
        }, filename)
        print(f"💾 Bagging модель сохранена: {filename}")

    def load_model(self, filename):
        """
        Загрузка модели
        """
        loaded = joblib.load(filename)
        self.model = loaded['model']
        self.vectorizer = loaded['vectorizer']
        self.base_estimator = loaded.get('base_estimator', 'logistic')
        self.positive_label = loaded.get('positive_label', 1)
        self.negative_label = loaded.get('negative_label', 0)
        self.is_trained = True
        print(f"📥 Bagging модель загружена: {filename}")


# Сравнение разных конфигураций Bagging
def compare_bagging_configs(train_data, val_data):
    """
    Сравнение разных конфигураций Bagging
    """
    print("🔬 СРАВНЕНИЕ КОНФИГУРАЦИЙ BAGGING")
    print("=" * 50)

    models = {}

    # 1. Bagging с логистической регрессией
    configs = [
        ('logistic', 10, 0.8, 0.8),
        ('logistic', 20, 0.8, 0.8),
        ('logistic', 10, 1.0, 0.6),
    ]

    for base_est, n_est, max_samp, max_feat in configs:
        print(f"\n1. Bagging {base_est} (n_est={n_est}, samples={max_samp}, features={max_feat}):")
        model = BaggingSentimentClassifier(
            base_estimator=base_est,
            n_estimators=n_est,
            max_samples=max_samp,
            max_features=max_feat
        )
        model.train(train_data, val_data)
        models[f'Bagging_{base_est}_{n_est}'] = model

    # 2. Bagging с деревьями решений
    print(f"\n2. Bagging с Decision Trees:")
    model_tree = BaggingSentimentClassifier(
        base_estimator='tree',
        n_estimators=15,
        max_samples=0.7,
        max_features=0.7
    )
    model_tree.train(train_data, val_data)
    models['Bagging_tree'] = model_tree

    return models


# Анализ стабильности ансамбля
def analyze_ensemble_stability(model, data):
    """
    Анализ стабильности и согласованности ансамбля
    """
    print(f"\n📊 АНАЛИЗ СТАБИЛЬНОСТИ АНСАМБЛЯ:")

    # Предсказания с консенсусом
    results = model.predict_with_consensus([item['text'] for item in data])

    unanimous_count = sum(1 for r in results if r['unanimous'])
    high_consensus = sum(1 for r in results if r['consensus_ratio'] >= 0.8)

    print(f"   Единогласные решения: {unanimous_count}/{len(results)} ({unanimous_count / len(results) * 100:.1f}%)")
    print(f"   Высокий консенсус (≥80%): {high_consensus}/{len(results)} ({high_consensus / len(results) * 100:.1f}%)")

    # Разнообразие ансамбля
    diversity = model.get_ensemble_diversity(data)
    print(f"   Score разнообразия: {diversity['diversity_score']:.3f}")
    print(f"   Среднее несогласие: {diversity['average_disagreement']:.1f} пар на модель")


# Пример использования
def main():
    """
    Пример использования Bagging классификатора
    """
    train_data = read_jsonl_basic('../../util/news_sentiment_train.jsonl')
    val_data = read_jsonl_basic('../../util/news_sentiment_val.jsonl')
    test_data = read_jsonl_basic('../../util/news_sentiment_test.jsonl')

    print(f"📊 Данные: {len(train_data)} train, {len(val_data)} val")

    # Обучаем Bagging с логистической регрессией
    print("\n" + "=" * 50)
    bagging_classifier = BaggingSentimentClassifier(
        base_estimator='logistic',
        n_estimators=15,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True
    )
    bagging_classifier.train(train_data, val_data)

    analyze_ensemble_stability(bagging_classifier, val_data)

    bagging_classifier.save_model("bagging_sentiment_classifier.pkl")

    print("\n" + "=" * 50)
    models = compare_bagging_configs(train_data[:150], val_data[:30])


if __name__ == "__main__":
    main()