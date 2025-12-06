from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import json


class BaggingMultiClassClassifier:
    """
    Многоклассовый классификатор на основе Bagging ансамбля
    """

    def __init__(self, base_estimator='logistic', n_estimators=10,
                 max_samples=1.0, max_features=1.0, bootstrap=True,
                 bootstrap_features=False, random_state=42):
        """
        Args:
            base_estimator: 'logistic' или 'tree' - базовый алгоритм
            n_estimators: количество базовых классификаторов
            max_samples: доля/количество samples для каждого классификатора
            max_features: доля/количество features для каждого классификатора
            bootstrap: использовать ли bootstrap sampling
            bootstrap_features: использовать ли bootstrap для features
            random_state: для воспроизводимости
        """
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )

        # Для кодирования меток
        self.label_encoder = LabelEncoder()

        # Выбор базового классификатора
        if base_estimator == 'logistic':
            base_est = LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                C=1.0,
                multi_class='ovr'  # Для многоклассовой классификации
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
            n_jobs=-1,
            verbose=0
        )

        self.base_estimator = base_estimator
        self.is_trained = False
        self.class_names = None
        self.n_classes = None

    def prepare_data(self, data):
        """
        Подготовка данных: извлекаем тексты и метки
        """
        texts = [item['text'] for item in data]
        labels = [item['category'] for item in data]
        return texts, labels

    def train(self, train_data, val_data=None):
        """
        Обучение Bagging модели для многоклассовой классификации
        """
        print(f"🎯 ОБУЧЕНИЕ BAGGING ({self.base_estimator.upper()})...")

        # Подготовка данных
        X_train, y_train = self.prepare_data(train_data)

        # Кодируем метки
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.class_names = self.label_encoder.classes_
        self.n_classes = len(self.class_names)

        # Проверяем количество классов
        unique_labels = set(y_train)
        print(f"📊 Обнаружено {self.n_classes} классов: {list(self.class_names)}")

        # Векторизация текстов
        print("📊 Векторизация текстов...")
        X_train_vec = self.vectorizer.fit_transform(X_train)

        print(f"   Размерность признаков: {X_train_vec.shape}")
        print(f"   Количество классов: {self.n_classes}")
        print(f"   Количество примеров: {len(y_train)}")
        print(f"   Базовый алгоритм: {self.base_estimator}")
        print(f"   Количество моделей: {self.model.n_estimators}")
        print(f"   Max samples: {self.model.max_samples}")
        print(f"   Max features: {self.model.max_features}")
        print(f"   Bootstrap: {self.model.bootstrap}")
        print(f"   Bootstrap features: {self.model.bootstrap_features}")

        # Обучение модели
        print("🤖 Обучение Bagging ансамбля...")
        self.model.fit(X_train_vec, y_train_encoded)
        self.is_trained = True

        # Оценка на тренировочных данных
        train_pred = self.model.predict(X_train_vec)
        train_pred_decoded = self.label_encoder.inverse_transform(train_pred)
        train_accuracy = accuracy_score(y_train, train_pred_decoded)
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
        if self.base_estimator == 'logistic':
            self._show_important_features(top_n=10)

    def predict(self, texts):
        """
        Предсказание для списка текстов
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)

        # Получаем предсказания и вероятности
        predictions_encoded = self.model.predict(X_vec)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        probabilities = self.model.predict_proba(X_vec)

        return predictions, probabilities

    def predict_single(self, text):
        """
        Предсказание для одного текста с детальной информацией
        """
        predictions, probabilities = self.predict([text])
        pred = predictions[0]
        pred_encoded = self.label_encoder.transform([pred])[0]
        prob = probabilities[0]

        # Получаем вероятности для всех классов
        class_probabilities = {}
        for class_name, prob_value in zip(self.class_names, prob):
            class_probabilities[class_name] = prob_value

        # Находим наиболее вероятные классы
        sorted_probs = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_probs[:3]

        return {
            'prediction': pred,
            'confidence': float(prob[pred_encoded]),
            'probabilities': class_probabilities,
            'top_3_predictions': top_3,
            'model_type': type(self.model.estimator).__name__,
            'ensemble_size': self.model.n_estimators
        }

    def predict_with_voting_details(self, texts):
        """
        Предсказание с детальной информацией о голосовании ансамбля
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)

        # Матрица голосов: [n_estimators, n_samples]
        all_predictions = []

        for i, estimator in enumerate(self.model.estimators_):
            try:
                # Учитываем подмножества признаков, если используются
                if hasattr(self.model, 'estimators_features_') and self.model.estimators_features_:
                    features_idx = self.model.estimators_features_[i]
                    X_subset = X_vec[:, features_idx]
                else:
                    X_subset = X_vec

                predictions_encoded = estimator.predict(X_subset)
                all_predictions.append(predictions_encoded)
            except Exception as e:
                print(f"⚠️  Ошибка в модели {i}: {e}")
                continue

        if not all_predictions:
            raise Exception("Ни одна модель не смогла сделать предсказание")

        all_predictions = np.array(all_predictions)  # [n_estimators, n_samples]
        n_estimators, n_samples = all_predictions.shape

        # Основное предсказание
        final_predictions_encoded = self.model.predict(X_vec)
        final_predictions = self.label_encoder.inverse_transform(final_predictions_encoded)
        probabilities = self.model.predict_proba(X_vec)

        results = []
        for i in range(n_samples):
            # Голоса для каждого класса
            votes = {}
            for class_idx, class_name in enumerate(self.class_names):
                votes[class_name] = np.sum(all_predictions[:, i] == class_idx)

            # Считаем распределение голосов
            total_votes = n_estimators
            sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            consensus_ratio = sorted_votes[0][1] / total_votes

            # Информация о победившем классе
            winner_class = final_predictions[i]
            winner_votes = votes[winner_class]

            results.append({
                'prediction': winner_class,
                'probability': probabilities[i],
                'votes': votes,
                'total_votes': total_votes,
                'winner_votes': winner_votes,
                'consensus_ratio': consensus_ratio,
                'unanimous': consensus_ratio == 1.0,
                'vote_distribution': sorted_votes
            })

        return results

    def evaluate(self, test_data):
        """
        Оценка модели на тестовых данных для многоклассовой классификации
        """
        X_test, y_test = self.prepare_data(test_data)
        X_test_vec = self.vectorizer.transform(X_test)

        # Кодируем тестовые метки
        y_test_encoded = self.label_encoder.transform(y_test)

        y_pred_encoded = self.model.predict(X_test_vec)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)

        print("\n📊 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        print(classification_report(y_test, y_pred, digits=4))

        # Матрица ошибок
        print("\n📈 МАТРИЦА ОШИБОК:")
        cm = confusion_matrix(y_test_encoded, y_pred_encoded)

        # Используем актуальные имена классов из предсказаний
        unique_classes_in_test = np.unique(y_pred)
        actual_class_names = [str(cls) for cls in unique_classes_in_test]

        # Если в тесте нет всех классов, используем только те, что есть
        if len(actual_class_names) < self.n_classes:
            print(f"⚠️  В тестовых данных не все классы. Представлено: {len(actual_class_names)} из {self.n_classes}")
            self._print_confusion_matrix_simple(cm, actual_class_names)
        else:
            self._print_confusion_matrix(cm, self.class_names)

        # Дополнительные метрики
        print(f"\n📊 ОБЩИЕ МЕТРИКИ:")
        print(f"   Accuracy: {accuracy:.4f}")

        # Средние метрики
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision_macro = precision_score(y_test_encoded, y_pred_encoded, average='macro')
        recall_macro = recall_score(y_test_encoded, y_pred_encoded, average='macro')
        f1_macro = f1_score(y_test_encoded, y_pred_encoded, average='macro')

        precision_weighted = precision_score(y_test_encoded, y_pred_encoded, average='weighted')
        recall_weighted = recall_score(y_test_encoded, y_pred_encoded, average='weighted')
        f1_weighted = f1_score(y_test_encoded, y_pred_encoded, average='weighted')

        print(f"   Precision (macro): {precision_macro:.4f}")
        print(f"   Recall (macro): {recall_macro:.4f}")
        print(f"   F1-Score (macro): {f1_macro:.4f}")
        print(f"   Precision (weighted): {precision_weighted:.4f}")
        print(f"   Recall (weighted): {recall_weighted:.4f}")
        print(f"   F1-Score (weighted): {f1_weighted:.4f}")

        return accuracy

    def _print_confusion_matrix(self, cm, class_names):
        """
        Красивый вывод матрицы ошибок
        """
        n_classes = len(class_names)

        # Проверяем размерность матрицы
        if cm.shape[0] != n_classes:
            print(f"⚠️  Размер матрицы ({cm.shape[0]}) не совпадает с количеством классов ({n_classes})")
            # Обрезаем или дополняем имена классов до размера матрицы
            if cm.shape[0] < n_classes:
                class_names = class_names[:cm.shape[0]]
            n_classes = cm.shape[0]

        # Заголовок
        header = " " * 15 + "Предсказано →"
        print(header)

        # Имена классов для предсказаний
        pred_header = " " * 10
        for name in class_names[:n_classes]:
            pred_header += f"{str(name)[:8]:^8} "
        print(pred_header)

        # Разделитель
        separator = " " * 10 + "─" * (n_classes * 9)
        print(separator)

        # Строки матрицы
        for i, true_name in enumerate(class_names[:n_classes]):
            row = f"Истинно {str(true_name)[:8]:<8}│"
            for j in range(n_classes):
                row += f"{cm[i][j]:^8} "
            print(row)

        # Статистика
        diagonal = cm.diagonal()
        total = cm.sum()
        correct = diagonal.sum()

        print(f"\n📊 Статистика:")
        print(f"   Правильно классифицировано: {correct}/{total} ({correct / total:.1%})")

    def _print_confusion_matrix_simple(self, cm, class_names):
        """
        Упрощенный вывод матрицы ошибок
        """
        n_classes = len(class_names)

        print("\n      Предсказано")
        print("      " + " ".join([f"{name[:4]:>4}" for name in class_names]))
        print("     ┌" + "─" * (n_classes * 5 + n_classes - 1) + "┐")

        for i, true_name in enumerate(class_names):
            row = f"{true_name[:4]:>4} │"
            for j in range(n_classes):
                row += f" {cm[i][j]:>3}"
            row += " │"
            print(row)

            if i < n_classes - 1:
                print("     ├" + "─" * (n_classes * 5 + n_classes - 1) + "┤")

        print("     └" + "─" * (n_classes * 5 + n_classes - 1) + "┘")

        # Статистика
        diagonal = cm.diagonal()
        total = cm.sum()
        correct = diagonal.sum()

        print(f"\n📊 Статистика:")
        print(f"   Правильно классифицировано: {correct}/{total} ({correct / total:.1%})")

    def _analyze_ensemble(self):
        """
        Анализ разнообразия и качества ансамбля
        """
        print(f"\n📊 АНАЛИЗ BAGGING АНСАМБЛЯ:")
        print(f"   Количество моделей: {len(self.model.estimators_)}")
        print(f"   Количество классов: {self.n_classes}")

        if hasattr(self.model, 'estimators_features_'):
            unique_features_sets = len(set(
                tuple(sorted(features)) for features in self.model.estimators_features_
            ))
            print(f"   Уникальных наборов признаков: {unique_features_sets}")

        # Out-of-bag score
        if hasattr(self.model, 'oob_score_'):
            print(f"   Out-of-bag score: {self.model.oob_score_:.3f}")

    def _show_important_features(self, top_n=10):
        """
        Показывает важные признаки для каждого класса (для логистической регрессии)
        """
        if self.base_estimator != 'logistic':
            print(f"\n⚠️  Важность признаков недоступна для базового {self.base_estimator}")
            return

        try:
            # Для многоклассовой логистической регрессии коэффициенты: [n_classes, n_features]
            all_coefs = []
            for estimator in self.model.estimators_:
                if hasattr(estimator, 'coef_'):
                    all_coefs.append(estimator.coef_)  # [n_classes, n_features]

            if not all_coefs:
                print("❌ Не удалось получить коэффициенты моделей")
                return

            # Усредненные коэффициенты по всем моделям ансамбля
            avg_coefs = np.mean(all_coefs, axis=0)  # [n_classes, n_features]
            feature_names = self.vectorizer.get_feature_names_out()

            print(f"\n🔍 ВАЖНЫЕ ПРИЗНАКИ ДЛЯ КАЖДОГО КЛАССА:")

            for class_idx, class_name in enumerate(self.class_names):
                print(f"\n   🎯 КЛАСС: {class_name}")

                # Положительные признаки для этого класса
                class_coefs = avg_coefs[class_idx]

                # Топ положительных признаков (указывают на этот класс)
                pos_indices = np.argsort(class_coefs)[-top_n:][::-1]
                if len(pos_indices) > 0:
                    print(f"      Топ положительных признаков:")
                    for idx in pos_indices[:min(top_n, len(pos_indices))]:
                        print(f"        {feature_names[idx]}: {class_coefs[idx]:.3f}")

                # Топ отрицательных признаков (против этого класса)
                neg_indices = np.argsort(class_coefs)[:top_n]
                if len(neg_indices) > 0:
                    print(f"      Топ отрицательных признаков:")
                    for idx in neg_indices[:min(top_n, len(neg_indices))]:
                        print(f"        {feature_names[idx]}: {class_coefs[idx]:.3f}")

        except Exception as e:
            print(f"❌ Ошибка при анализе важности признаков: {e}")

    def get_ensemble_diversity(self, data):
        """
        Оценивает разнообразие ансамбля для многоклассовой классификации
        """
        X, y = self.prepare_data(data)
        X_vec = self.vectorizer.transform(X)
        y_encoded = self.label_encoder.transform(y)

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

        all_predictions = np.array(all_predictions)  # [n_models, n_samples]
        n_models, n_samples = all_predictions.shape

        # Считаем попарные различия
        disagreements = 0
        total_pairs = 0

        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreements += np.sum(all_predictions[i] != all_predictions[j])
                total_pairs += n_samples

        diversity_score = disagreements / total_pairs if total_pairs > 0 else 0

        return {
            'diversity_score': diversity_score,
            'average_disagreement': disagreements / (n_models * (n_models - 1) / 2) if n_models > 1 else 0,
            'n_models': n_models,
            'n_classes': self.n_classes
        }

    def get_class_distribution(self, data):
        """
        Возвращает распределение классов в данных
        """
        _, labels = self.prepare_data(data)
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))

    def save_model(self, filename):
        """
        Сохранение модели
        """
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'base_estimator': self.base_estimator,
            'class_names': self.class_names,
            'n_classes': self.n_classes
        }, filename)
        print(f"💾 Bagging многоклассовая модель сохранена: {filename}")

    def load_model(self, filename):
        """
        Загрузка модели
        """
        loaded = joblib.load(filename)
        self.model = loaded['model']
        self.vectorizer = loaded['vectorizer']
        self.label_encoder = loaded['label_encoder']
        self.base_estimator = loaded.get('base_estimator', 'logistic')
        self.class_names = loaded.get('class_names', [])
        self.n_classes = loaded.get('n_classes', 0)
        self.is_trained = True
        print(f"📥 Bagging многоклассовая модель загружена: {filename}")


# СПЕЦИАЛИЗИРОВАННЫЙ КЛАССИФИКАТОР ДЛЯ ТОНАЛЬНОСТИ (3 КЛАССА)
class SentimentBaggingClassifier(BaggingMultiClassClassifier):
    """
    Специализированный Bagging классификатор для тональности (негатив, нейтрал, позитив)
    """

    def __init__(self, base_estimator='logistic', n_estimators=15,
                 max_samples=0.8, max_features=0.8, bootstrap=True,
                 random_state=42):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state
        )

    def predict_sentiment(self, text):
        """
        Специализированный метод для предсказания тональности
        """
        result = super().predict_single(text)

        # Добавляем интерпретацию для тональности
        sentiment_mapping = {
            'negative': 'негативный',
            'neutral': 'нейтральный',
            'positive': 'позитивный',
            'neg': 'негативный',
            'neu': 'нейтральный',
            'pos': 'позитивный',
            '0': 'негативный',
            '1': 'нейтральный',
            '2': 'позитивный'
        }

        prediction = result['prediction']
        sentiment = sentiment_mapping.get(str(prediction).lower(), prediction)

        result['sentiment'] = sentiment
        result['sentiment_confidence'] = result['confidence']

        # Детали голосования для тональности
        voting_details = self.predict_with_voting_details([text])[0]
        result['voting_details'] = voting_details

        return result


# СРАВНЕНИЕ РАЗНЫХ КОНФИГУРАЦИЙ BAGGING
def compare_bagging_configs_multiclass(train_data, val_data):
    """
    Сравнение разных конфигураций Bagging для многоклассовой классификации
    """
    print("🔬 СРАВНЕНИЕ КОНФИГУРАЦИЙ BAGGING (многоклассовый)")
    print("=" * 50)

    models = {}

    # 1. Bagging с логистической регрессией
    configs = [
        ('logistic', 10, 0.8, 0.8),
        ('logistic', 20, 0.8, 0.8),
        ('logistic', 10, 0.6, 0.6),
    ]

    for base_est, n_est, max_samp, max_feat in configs:
        print(f"\n1. Bagging {base_est} (n_est={n_est}, samples={max_samp}, features={max_feat}):")
        model = BaggingMultiClassClassifier(
            base_estimator=base_est,
            n_estimators=n_est,
            max_samples=max_samp,
            max_features=max_feat
        )
        # Используем try-except для обработки ошибок
        try:
            model.train(train_data, val_data)
            models[f'Bagging_{base_est}_{n_est}'] = model
        except Exception as e:
            print(f"   ❌ Ошибка при обучении: {e}")
            continue

    # 2. Bagging с деревьями решений
    print(f"\n2. Bagging с Decision Trees:")
    model_tree = BaggingMultiClassClassifier(
        base_estimator='tree',
        n_estimators=15,
        max_samples=0.7,
        max_features=0.7
    )
    try:
        model_tree.train(train_data, val_data)
        models['Bagging_tree'] = model_tree
    except Exception as e:
        print(f"   ❌ Ошибка при обучении: {e}")

    return models


# АНАЛИЗ СТАБИЛЬНОСТИ АНСАМБЛЯ
def analyze_ensemble_stability_multiclass(model, data):
    """
    Анализ стабильности и согласованности ансамбля для многоклассовой классификации
    """
    print(f"\n📊 АНАЛИЗ СТАБИЛЬНОСТИ АНСАМБЛЯ (многоклассовый):")

    # Используем только часть данных для анализа (чтобы не перегружать)
    sample_data = data[:20] if len(data) > 20 else data

    # Предсказания с деталями голосования
    try:
        results = model.predict_with_voting_details([item['text'] for item in sample_data])
    except Exception as e:
        print(f"⚠️  Ошибка при анализе стабильности: {e}")
        return

    unanimous_count = sum(1 for r in results if r['unanimous'])
    high_consensus = sum(1 for r in results if r['consensus_ratio'] >= 0.8)
    low_consensus = sum(1 for r in results if r['consensus_ratio'] < 0.5)

    print(f"   Единогласные решения: {unanimous_count}/{len(results)} ({unanimous_count / len(results) * 100:.1f}%)")
    print(f"   Высокий консенсус (≥80%): {high_consensus}/{len(results)} ({high_consensus / len(results) * 100:.1f}%)")
    print(f"   Низкий консенсус (<50%): {low_consensus}/{len(results)} ({low_consensus / len(results) * 100:.1f}%)")

    # Распределение голосов по классам
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ ГОЛОСОВ:")
    all_votes = {}
    for class_name in model.class_names:
        all_votes[class_name] = 0

    for result in results:
        for class_name, votes in result['votes'].items():
            all_votes[class_name] += votes

    total_votes = sum(all_votes.values())
    for class_name, votes in all_votes.items():
        if total_votes > 0:
            percentage = (votes / total_votes) * 100
            print(f"   {class_name}: {votes} голосов ({percentage:.1f}%)")

    # Разнообразие ансамбля
    diversity = model.get_ensemble_diversity(sample_data)
    print(f"\n📊 РАЗНООБРАЗИЕ АНСАМБЛЯ:")
    print(f"   Score разнообразия: {diversity['diversity_score']:.3f}")
    if diversity['n_models'] > 1:
        print(f"   Среднее несогласие: {diversity['average_disagreement']:.1f} пар на модель")


# ПРИМЕР ИСПОЛЬЗОВАНИЯ
def main():
    """
    Пример использования многоклассового Bagging классификатора
    """

    # Функция для загрузки данных
    def load_jsonl(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    train_data = load_jsonl('../util/news_category_train.jsonl')
    test_data = load_jsonl('../util/news_category_test.jsonl')

    print(f"📊 Обучающих примеров: {len(train_data)}")
    print(f"📊 Тестовых примеров: {len(test_data)}")

    # Разделяем на train/val
    np.random.seed(42)
    indices = np.random.permutation(len(train_data))
    split_idx = int(0.8 * len(train_data))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_subset = [train_data[i] for i in train_indices]
    val_subset = [train_data[i] for i in val_indices]

    print(f"📊 Train: {len(train_subset)} примеров")
    print(f"📊 Val: {len(val_subset)} примеров")

    # Создаем и обучаем Bagging с логистической регрессией
    print("\n" + "=" * 50)
    print("🎯 ОБУЧЕНИЕ BAGGING (многоклассовый)")
    print("=" * 50)

    bagging_classifier = BaggingMultiClassClassifier(
        base_estimator='logistic',
        n_estimators=10,  # Уменьшено для скорости
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True
    )

    bagging_classifier.train(train_subset, val_subset)

    # Анализ стабильности ансамбля
    analyze_ensemble_stability_multiclass(bagging_classifier, val_subset[:20])

    # Оценка на тестовых данных
    print("\n🧪 ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ...")
    test_accuracy = bagging_classifier.evaluate(test_data)

    # Сохранение модели
    bagging_classifier.save_model("bagging_multiclass_model.pkl")

    # Загружаем модель и тестируем
    print("\n🧪 ТЕСТ ЗАГРУЗКИ МОДЕЛИ...")
    loaded_classifier = BaggingMultiClassClassifier()
    loaded_classifier.load_model("bagging_multiclass_model.pkl")

    # Быстрый тест загруженной модели
    test_text = "Довольно неплохо, но есть небольшие замечания"
    result = loaded_classifier.predict_single(test_text)
    print(f"\n📝 Тест загруженной модели:")
    print(f"   Текст: {test_text}")
    print(f"   Предсказание: {result['prediction']}")
    print(f"   Уверенность: {result['confidence']:.3f}")

    # Сравнение конфигураций (если есть достаточно данных)
    print("\n" + "=" * 50)
    print("🔬 СРАВНЕНИЕ КОНФИГУРАЦИЙ")
    print("=" * 50)

    # Используем меньше данных для быстрого сравнения
    small_train = train_subset[:60]
    small_val = val_subset[:15]

    # Проверяем, есть ли достаточно данных
    if len(small_train) >= 10 and len(small_val) >= 5:
        models = compare_bagging_configs_multiclass(small_train, small_val)
    else:
        print("⚠️  Недостаточно данных для сравнения конфигураций")

    return bagging_classifier


if __name__ == "__main__":
    print("🚀 ЗАПУСК МНОГОКЛАССОВОГО BAGGING КЛАССИФИКАТОРА")
    print("=" * 80)

    # Запускаем основной пример
    classifier = main()