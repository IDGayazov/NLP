from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import joblib

from util.jsonl_process import read_jsonl_basic


class VotingSentimentClassifier:
    """
    Классификатор тональности на основе голосования (Voting)
    """

    def __init__(self, voting_type='soft', positive_label=1, negative_label=0, random_state=42):
        """
        Args:
            voting_type: 'hard' или 'soft' голосование
            positive_label: метка положительного класса
            negative_label: метка отрицательного класса
            random_state: для воспроизводимости
        """
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            min_df=2,
            max_df=0.85,
            ngram_range=(1, 2),
            stop_words=None
        )

        # Создаем разнообразные модели для голосования
        self.models = {
            'logistic': LogisticRegression(
                C=1.0,
                random_state=random_state,
                max_iter=1000
            ),
            'svm': SVC(
                C=1.0,
                kernel='linear',
                probability=True,  # Для soft voting
                random_state=random_state
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=random_state
            ),
            'svm_rbf': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=random_state
            ),
            'logistic_l2': LogisticRegression(
                C=0.1,
                penalty='l2',
                random_state=random_state,
                max_iter=1000
            )
        }

        self.voting_classifier = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting=voting_type,
            n_jobs=-1
        )

        self.voting_type = voting_type
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.is_trained = False
        self.random_state = random_state

    def prepare_data(self, data):
        """
        Подготовка данных: извлекаем тексты и метки
        """
        texts = [item['text'] for item in data]
        labels = [item['sentiment'] for item in data]
        return texts, labels

    def train(self, train_data, val_data=None):
        """
        Обучение voting классификатора
        """
        print(f"🎯 ОБУЧЕНИЕ {self.voting_type.upper()} VOTING КЛАССИФИКАТОРА...")

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
        print(f"   Тип голосования: {self.voting_type}")
        print(f"   Модели в ансамбле: {list(self.models.keys())}")
        print(f"   Количество моделей: {len(self.models)}")

        # Обучение voting классификатора
        print("\n🤖 Обучение ансамбля...")
        self.voting_classifier.fit(X_train_vec, y_train)
        self.is_trained = True

        # Оценка на тренировочных данных
        train_pred = self.voting_classifier.predict(X_train_vec)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"✅ Точность на train: {train_accuracy:.3f}")

        # Оценка индивидуальных моделей
        self._evaluate_individual_models(X_train_vec, y_train)

        # Оценка на валидации, если есть
        if val_data:
            val_accuracy = self.evaluate(val_data)
            print(f"✅ Точность на val: {val_accuracy:.3f}")

        # Анализ ансамбля
        self._analyze_ensemble(X_train_vec, y_train)

    def _evaluate_individual_models(self, X_vec, y_true):
        """
        Оценка индивидуальных моделей
        """
        print(f"\n📊 ПРОИЗВОДИТЕЛЬНОСТЬ ИНДИВИДУАЛЬНЫХ МОДЕЛЕЙ:")
        print("-" * 50)

        individual_accuracies = {}

        for name, model in self.models.items():
            try:
                # Обучаем модель отдельно если она еще не обучена
                if not hasattr(model, 'classes_'):
                    model.fit(X_vec, y_true)

                pred = model.predict(X_vec)
                accuracy = accuracy_score(y_true, pred)
                individual_accuracies[name] = accuracy
                print(f"   {name:<15}: {accuracy:.3f}")

            except Exception as e:
                print(f"   {name:<15}: ошибка - {e}")
                individual_accuracies[name] = 0

        return individual_accuracies

    def _analyze_ensemble(self, X_vec, y_true):
        """
        Анализ работы ансамбля
        """
        print(f"\n📊 АНАЛИЗ {self.voting_type.upper()} VOTING АНСАМБЛЯ:")
        print("-" * 50)

        # Получаем предсказания всех моделей
        all_predictions = {}
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                all_predictions[name] = model.predict(X_vec)

        # Анализ согласованности
        n_samples = len(y_true)
        unanimous_count = 0
        high_agreement_count = 0

        for i in range(n_samples):
            votes = [pred[i] for pred in all_predictions.values()]
            positive_votes = sum(1 for v in votes if v == self.positive_label)
            negative_votes = sum(1 for v in votes if v == self.negative_label)

            if positive_votes == len(votes) or negative_votes == len(votes):
                unanimous_count += 1
            if max(positive_votes, negative_votes) >= len(votes) * 0.8:
                high_agreement_count += 1

        print(f"   Единогласные решения: {unanimous_count}/{n_samples} ({unanimous_count / n_samples * 100:.1f}%)")
        print(
            f"   Высокое согласие (≥80%): {high_agreement_count}/{n_samples} ({high_agreement_count / n_samples * 100:.1f}%)")

        # Анализ разнообразия
        if len(all_predictions) > 1:
            diversity = self._calculate_diversity(all_predictions)
            print(f"   Разнообразие ансамбля: {diversity:.3f}")

    def _calculate_diversity(self, predictions_dict):
        """
        Вычисляет разнообразие ансамбля
        """
        predictions = list(predictions_dict.values())
        n_models = len(predictions)
        n_samples = len(predictions[0])

        disagreements = 0
        total_pairs = 0

        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreements += np.sum(predictions[i] != predictions[j])
                total_pairs += n_samples

        return disagreements / total_pairs if total_pairs > 0 else 0

    def predict(self, texts):
        """
        Предсказание для списка текстов
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)
        predictions = self.voting_classifier.predict(X_vec)

        # Для soft voting получаем вероятности, для hard voting - вычисляем
        if self.voting_type == 'soft':
            probabilities = self.voting_classifier.predict_proba(X_vec)
        else:
            probabilities = self._get_hard_voting_probabilities(X_vec)

        return predictions, probabilities

    def _get_hard_voting_probabilities(self, X_vec):
        """
        Вычисляет псевдо-вероятности для hard voting
        """
        # Получаем предсказания всех моделей
        all_predictions = []
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                pred = model.predict(X_vec)
                all_predictions.append(pred)

        if not all_predictions:
            raise Exception("Не удалось получить предсказания моделей")

        all_predictions = np.array(all_predictions)
        n_models = len(all_predictions)
        n_samples = len(all_predictions[0])

        # Вычисляем вероятности на основе голосования
        probabilities = np.zeros((n_samples, 2))

        for i in range(n_samples):
            votes = all_predictions[:, i]
            positive_votes = np.sum(votes == self.positive_label)
            negative_votes = np.sum(votes == self.negative_label)

            probabilities[i, 0] = negative_votes / n_models  # Вероятность отрицательного
            probabilities[i, 1] = positive_votes / n_models  # Вероятность положительного

        return probabilities

    def predict_single(self, text):
        """
        Предсказание для одного текста с детальной информацией
        """
        predictions, probabilities = self.predict([text])
        pred = predictions[0]
        prob = probabilities[0]

        # Определяем вероятности для каждого класса
        if self.voting_classifier.classes_[0] == self.positive_label:
            pos_prob = prob[0]
            neg_prob = prob[1]
        else:
            pos_prob = prob[1]
            neg_prob = prob[0]

        sentiment = "POSITIVE" if pred == self.positive_label else "NEGATIVE"

        # Получаем детальную информацию о голосовании
        voting_details = self._get_voting_details(text)

        return {
            'prediction': pred,
            'sentiment': sentiment,
            'positive_prob': pos_prob,
            'negative_prob': neg_prob,
            'confidence': max(pos_prob, neg_prob),
            'voting_type': self.voting_type,
            'voting_details': voting_details
        }

    def _get_voting_details(self, text):
        """
        Получает детальную информацию о голосовании моделей
        """
        X_vec = self.vectorizer.transform([text])

        voting_results = {}
        all_predictions = []
        all_probabilities = []

        for name, model in self.models.items():
            try:
                pred = model.predict(X_vec)[0]

                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_vec)[0]
                else:
                    # Для моделей без вероятностей создаем равномерное распределение
                    prob = np.array([0.5, 0.5]) if pred == self.positive_label else np.array([0.5, 0.5])

                voting_results[name] = {
                    'prediction': pred,
                    'probability': prob,
                    'sentiment': "POSITIVE" if pred == self.positive_label else "NEGATIVE",
                    'confidence': max(prob)
                }

                all_predictions.append(pred)
                all_probabilities.append(prob)

            except Exception as e:
                print(f"⚠️  Ошибка в модели {name}: {e}")
                continue

        # Анализ голосования
        positive_votes = sum(1 for p in all_predictions if p == self.positive_label)
        negative_votes = sum(1 for p in all_predictions if p == self.negative_label)
        total_votes = len(all_predictions)

        if self.voting_type == 'soft':
            # Для soft voting вычисляем средние вероятности
            avg_probabilities = np.mean(all_probabilities, axis=0)
            soft_positive_prob = avg_probabilities[1] if self.voting_classifier.classes_[0] == self.positive_label else \
            avg_probabilities[0]
            decision_reason = f"Soft voting (средняя вероятность: {soft_positive_prob:.3f})"
        else:
            decision_reason = f"Hard voting ({positive_votes}/{total_votes} за POSITIVE)"

        return {
            'individual_votes': voting_results,
            'positive_votes': positive_votes,
            'negative_votes': negative_votes,
            'total_votes': total_votes,
            'consensus_ratio': max(positive_votes, negative_votes) / total_votes,
            'unanimous': positive_votes == total_votes or negative_votes == total_votes,
            'decision_reason': decision_reason
        }

    def evaluate(self, test_data):
        """
        Оценка модели на тестовых данных
        """
        X_test, y_test = self.prepare_data(test_data)
        X_test_vec = self.vectorizer.transform(X_test)

        y_pred = self.voting_classifier.predict(X_test_vec)
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

    def compare_with_individual_models(self, test_data):
        """
        Сравнение ансамбля с индивидуальными моделями
        """
        X_test, y_test = self.prepare_data(test_data)
        X_test_vec = self.vectorizer.transform(X_test)

        print(f"\n🔬 СРАВНЕНИЕ {self.voting_type.upper()} VOTING С ИНДИВИДУАЛЬНЫМИ МОДЕЛЯМИ:")
        print("=" * 60)

        # Точность ансамбля
        ensemble_pred = self.voting_classifier.predict(X_test_vec)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        print(f"   {'VOTING ENSEMBLE':<20}: {ensemble_accuracy:.3f}")

        # Точности индивидуальных моделей
        individual_accuracies = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X_test_vec)
                accuracy = accuracy_score(y_test, pred)
                individual_accuracies[name] = accuracy
                print(f"   {name:<20}: {accuracy:.3f}")
            except Exception as e:
                print(f"   {name:<20}: ошибка - {e}")

        # Анализ улучшения
        if individual_accuracies:
            best_individual = max(individual_accuracies.values())
            improvement = ensemble_accuracy - best_individual
            print(f"\n   📈 Улучшение над лучшей моделью: {improvement:.3f}")
            print(f"   📈 Относительное улучшение: {improvement / best_individual * 100:.1f}%")

            return {
                'ensemble_accuracy': ensemble_accuracy,
                'individual_accuracies': individual_accuracies,
                'improvement': improvement,
                'best_individual': best_individual
            }

    def save_model(self, filename):
        """
        Сохранение модели
        """
        joblib.dump({
            'voting_classifier': self.voting_classifier,
            'vectorizer': self.vectorizer,
            'models': self.models,
            'voting_type': self.voting_type,
            'positive_label': self.positive_label,
            'negative_label': self.negative_label
        }, filename)
        print(f"💾 Voting модель сохранена: {filename}")

    def load_model(self, filename):
        """
        Загрузка модели
        """
        loaded = joblib.load(filename)
        self.voting_classifier = loaded['voting_classifier']
        self.vectorizer = loaded['vectorizer']
        self.models = loaded['models']
        self.voting_type = loaded['voting_type']
        self.positive_label = loaded.get('positive_label', 1)
        self.negative_label = loaded.get('negative_label', 0)
        self.is_trained = True
        print(f"📥 Voting модель загружена: {filename}")


# Сравнение Hard vs Soft Voting
def compare_voting_strategies(train_data, val_data, test_data):
    """
    Сравнение Hard Voting и Soft Voting
    """
    print("🔬 СРАВНЕНИЕ HARD VS SOFT VOTING")
    print("=" * 50)

    results = {}

    for voting_type in ['hard', 'soft']:
        print(f"\n🎯 {voting_type.upper()} VOTING:")
        voting_classifier = VotingSentimentClassifier(voting_type=voting_type)
        voting_classifier.train(train_data, val_data)

        # Сравнение с индивидуальными моделями
        comparison = voting_classifier.compare_with_individual_models(test_data)

        results[voting_type] = {
            'classifier': voting_classifier,
            'comparison': comparison
        }

    # Итоговое сравнение
    print("\n📊 ИТОГОВОЕ СРАВНЕНИЕ:")
    print("=" * 30)
    for voting_type, result in results.items():
        accuracy = result['comparison']['ensemble_accuracy']
        improvement = result['comparison']['improvement']
        print(f"   {voting_type.upper()} Voting: {accuracy:.3f} (улучшение: {improvement:+.3f})")

    return results


# Пример использования
def main():
    """
    Пример использования Voting классификатора
    """
    train_data = read_jsonl_basic('../../util/news_sentiment_train.jsonl')
    val_data = read_jsonl_basic('../../util/news_sentiment_val.jsonl')
    test_data = read_jsonl_basic('../../util/news_sentiment_test.jsonl')

    print(f"📊 Данные: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Тестирование Soft Voting
    print("\n" + "=" * 50)
    soft_voting = VotingSentimentClassifier(voting_type='soft')
    soft_voting.train(train_data, val_data)

    # Тестирование Hard Voting
    print("\n" + "=" * 50)
    hard_voting = VotingSentimentClassifier(voting_type='hard')
    hard_voting.train(train_data, val_data)

    print("\n" + "=" * 50)
    results = compare_voting_strategies(train_data, val_data, test_data)

    soft_voting.save_model("soft_voting_classifier.pkl")
    hard_voting.save_model("hard_voting_classifier.pkl")


if __name__ == "__main__":
    main()