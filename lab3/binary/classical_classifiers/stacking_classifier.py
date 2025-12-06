from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
from catboost import CatBoostClassifier
import numpy as np
import joblib
import warnings

from util.jsonl_process import read_jsonl_basic

warnings.filterwarnings('ignore')


class StackingSentimentClassifier:
    """
    Стекинг классификатор тональности с комбинацией SVM, LR и CatBoost
    """

    def __init__(self, use_blending=True, meta_model='logistic',
                 positive_label=1, negative_label=0, random_state=42):
        """
        Args:
            use_blending: True для блендинга, False для стекинга
            meta_model: тип мета-модели ('logistic', 'svm', 'random_forest')
            positive_label: метка положительного класса
            negative_label: метка отрицательного класса
            random_state: для воспроизводимости
        """
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            min_df=2,
            max_df=0.85,
            ngram_range=(1, 3),  # Добавляем триграммы для сложных моделей
            stop_words=None
        )

        # Базовые модели (level-0)
        self.base_models = {
            'svm': LinearSVC(
                C=1.0,
                random_state=random_state,
                max_iter=2000,
                dual=True
            ),
            'logistic': LogisticRegression(
                C=1.0,
                random_state=random_state,
                max_iter=1000,
                solver='liblinear'
            ),
            'catboost': CatBoostClassifier(
                iterations=500,
                learning_rate=0.1,
                depth=6,
                random_seed=random_state,
                verbose=0,
                thread_count=-1
            )
        }

        # Мета-модель (level-1)
        if meta_model == 'logistic':
            self.meta_model = LogisticRegression(
                C=1.0,
                random_state=random_state,
                max_iter=1000
            )
        elif meta_model == 'svm':
            self.meta_model = LinearSVC(
                C=1.0,
                random_state=random_state,
                max_iter=1000
            )
        elif meta_model == 'random_forest':
            self.meta_model = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                max_depth=None
            )
        else:
            raise ValueError("meta_model должен быть 'logistic', 'svm' или 'random_forest'")

        self.use_blending = use_blending
        self.meta_model_type = meta_model
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.is_trained = False
        self.random_state = random_state

        # Для хранения предсказаний при блендинге
        self.base_predictions = {}
        self.base_probabilities = {}

    def prepare_data(self, data):
        """
        Подготовка данных: извлекаем тексты и метки
        """
        texts = [item['text'] for item in data]
        labels = [item['sentiment'] for item in data]
        return texts, labels

    def train_blending(self, train_data, val_data):
        """
        Обучение с блендингом (используем отдельный validation set)
        """
        print("🎯 ОБУЧЕНИЕ С БЛЕНДИНГОМ...")

        X_train, y_train = self.prepare_data(train_data)
        X_val, y_val = self.prepare_data(val_data)

        # Векторизация
        print("📊 Векторизация текстов...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_val_vec = self.vectorizer.transform(X_val)

        print(f"   Размерность признаков: {X_train_vec.shape}")
        print(f"   Базовые модели: {list(self.base_models.keys())}")
        print(f"   Мета-модель: {self.meta_model_type}")

        # Обучаем базовые модели на тренировочных данных
        base_val_predictions = []
        base_val_probabilities = []

        print("\n🤖 ОБУЧЕНИЕ БАЗОВЫХ МОДЕЛЕЙ:")
        for name, model in self.base_models.items():
            print(f"   Обучение {name}...")

            if name == 'catboost':
                # CatBoost требует плотные массивы
                X_train_dense = X_train_vec.toarray()
                X_val_dense = X_val_vec.toarray()
                model.fit(X_train_dense, y_train)

                # Предсказания на validation set
                val_pred = model.predict(X_val_dense)
                val_prob = model.predict_proba(X_val_dense)
            else:
                model.fit(X_train_vec, y_train)
                val_pred = model.predict(X_val_vec)

                # Для SVM без вероятностей используем decision function
                if hasattr(model, 'predict_proba'):
                    val_prob = model.predict_proba(X_val_vec)
                else:
                    decision_scores = model.decision_function(X_val_vec)
                    val_prob = self._decision_to_probability(decision_scores)

            accuracy = accuracy_score(y_val, val_pred)
            print(f"      ✅ Точность на val: {accuracy:.3f}")

            base_val_predictions.append(val_pred.reshape(-1, 1))
            base_val_probabilities.append(val_prob)

            # Сохраняем обученные модели
            self.base_predictions[name] = val_pred
            self.base_probabilities[name] = val_prob

        # Создаем мета-признаки из вероятностей
        meta_features = np.hstack(base_val_probabilities)

        print(f"\n📊 МЕТА-ПРИЗНАКИ:")
        print(f"   Размерность мета-признаков: {meta_features.shape}")
        print(f"   Обучение мета-модели на {len(y_val)} примерах...")

        # Обучаем мета-модель на предсказаниях базовых моделей
        self.meta_model.fit(meta_features, y_val)

        # Оценка мета-модели на validation set
        meta_pred = self.meta_model.predict(meta_features)
        meta_accuracy = accuracy_score(y_val, meta_pred)
        print(f"   ✅ Точность мета-модели на val: {meta_accuracy:.3f}")

        self.is_trained = True

        return {
            'base_accuracies': {name: accuracy_score(y_val, pred)
                                for name, pred in self.base_predictions.items()},
            'meta_accuracy': meta_accuracy
        }

    def train_stacking(self, train_data, val_data=None):
        """
        Обучение со стекингом (используем кросс-валидацию)
        """
        print("🎯 ОБУЧЕНИЕ СО СТЕКИНГОМ...")

        X_train, y_train = self.prepare_data(train_data)

        # Векторизация
        print("📊 Векторизация текстов...")
        X_train_vec = self.vectorizer.fit_transform(X_train)

        print(f"   Размерность признаков: {X_train_vec.shape}")
        print(f"   Базовые модели: {list(self.base_models.keys())}")
        print(f"   Мета-модель: {self.meta_model_type}")

        # Для мета-модели убедимся, что она поддерживает вероятности
        if self.meta_model_type == 'svm':
            # Заменяем LinearSVC на SVC с probability=True для стекинга
            from sklearn.svm import SVC
            meta_model = SVC(
                C=1.0,
                random_state=self.random_state,
                probability=True,  # Включаем вероятности
                kernel='linear'
            )
        else:
            meta_model = self.meta_model

        # Создаем стекинг классификатор
        estimators = [(name, model) for name, model in self.base_models.items()]

        self.stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_model,
            cv=3,
            passthrough=False,
            n_jobs=-1
        )

        print("\n🤖 ОБУЧЕНИЕ СТЕКИНГ МОДЕЛИ...")

        # Для CatBoost преобразуем в плотный формат
        if any(name == 'catboost' for name in self.base_models.keys()):
            X_train_dense = X_train_vec.toarray()
            self.stacking_model.fit(X_train_dense, y_train)
        else:
            self.stacking_model.fit(X_train_vec, y_train)

        self.is_trained = True

        # Оценка на тренировочных данных
        if any(name == 'catboost' for name in self.base_models.keys()):
            train_pred = self.stacking_model.predict(X_train_dense)
        else:
            train_pred = self.stacking_model.predict(X_train_vec)

        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"✅ Точность на train: {train_accuracy:.3f}")

        # Оценка на валидации, если есть
        if val_data:
            val_accuracy = self.evaluate(val_data)
            print(f"✅ Точность на val: {val_accuracy:.3f}")

        return train_accuracy

    def train(self, train_data, val_data=None):
        """
        Основной метод обучения
        """
        if self.use_blending:
            if val_data is None:
                raise ValueError("Для блендинга необходим validation set")
            return self.train_blending(train_data, val_data)
        else:
            return self.train_stacking(train_data, val_data)

    def predict(self, texts):
        """
        Предсказание для списка текстов
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)

        if self.use_blending:
            return self._predict_blending(X_vec)
        else:
            return self._predict_stacking(X_vec)

    def _predict_blending(self, X_vec):
        """
        Предсказание для блендинга
        """
        # Получаем предсказания от всех базовых моделей
        base_probabilities = []

        for name, model in self.base_models.items():
            if name == 'catboost':
                X_dense = X_vec.toarray()
                prob = model.predict_proba(X_dense)
            else:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_vec)
                else:
                    decision_scores = model.decision_function(X_vec)
                    prob = self._decision_to_probability(decision_scores)

            base_probabilities.append(prob)

        # Создаем мета-признаки
        meta_features = np.hstack(base_probabilities)

        # Предсказание мета-модели
        predictions = self.meta_model.predict(meta_features)
        probabilities = self._get_meta_probabilities(meta_features)

        return predictions, probabilities

    def _predict_stacking(self, X_vec):
        """
        Предсказание для стекинга
        """
        # Для CatBoost преобразуем в плотный формат
        if any(name == 'catboost' for name in self.base_models.keys()):
            X_dense = X_vec.toarray()
            predictions = self.stacking_model.predict(X_dense)
        else:
            predictions = self.stacking_model.predict(X_vec)

        # Получаем вероятности, если доступны
        if hasattr(self.stacking_model, 'predict_proba'):
            if any(name == 'catboost' for name in self.base_models.keys()):
                probabilities = self.stacking_model.predict_proba(X_dense)
            else:
                probabilities = self.stacking_model.predict_proba(X_vec)
        else:
            # Если predict_proba недоступен, создаем псевдо-вероятности
            if any(name == 'catboost' for name in self.base_models.keys()):
                decision_scores = self.stacking_model.decision_function(X_dense)
            else:
                decision_scores = self.stacking_model.decision_function(X_vec)
            probabilities = self._decision_to_probability(decision_scores)

        return predictions, probabilities

    def _decision_to_probability(self, decision_scores):
        """
        Преобразование decision function в вероятности для SVM
        """
        # Преобразуем в numpy array для гарантии
        decision_scores = np.array(decision_scores)

        # Простая сигмоидальная трансформация
        probabilities = 1 / (1 + np.exp(-decision_scores))
        prob_matrix = np.zeros((len(probabilities), 2))
        prob_matrix[:, 1] = probabilities
        prob_matrix[:, 0] = 1 - probabilities
        return prob_matrix

    def _get_meta_probabilities(self, meta_features):
        """
        Получение вероятностей от мета-модели
        """
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)
        else:
            # Для SVM без вероятностей
            decision_scores = self.meta_model.decision_function(meta_features)
            return self._decision_to_probability(decision_scores)

    def predict_single(self, text):
        """
        Предсказание для одного текста с детальной информацией
        """
        predictions, probabilities = self.predict([text])
        pred = predictions[0]
        prob = probabilities[0]

        # Определяем вероятности для каждого класса
        if self.meta_model.classes_[0] == self.positive_label:
            pos_prob = prob[0]
            neg_prob = prob[1]
        else:
            pos_prob = prob[1]
            neg_prob = prob[0]

        sentiment = "POSITIVE" if pred == self.positive_label else "NEGATIVE"

        # Получаем предсказания базовых моделей для анализа
        base_predictions = self._get_base_predictions(text)

        return {
            'prediction': pred,
            'sentiment': sentiment,
            'positive_prob': pos_prob,
            'negative_prob': neg_prob,
            'confidence': max(pos_prob, neg_prob),
            'base_predictions': base_predictions,
            'consensus': self._get_consensus(base_predictions)
        }

    def _get_base_predictions(self, text):
        """
        Получает предсказания всех базовых моделей
        """
        X_vec = self.vectorizer.transform([text])
        base_results = {}

        if self.use_blending:
            # Для блендинга - используем отдельно обученные модели
            for name, model in self.base_models.items():
                try:
                    if name == 'catboost':
                        X_dense = X_vec.toarray()
                        pred = model.predict(X_dense)[0]
                        prob = model.predict_proba(X_dense)[0]
                    else:
                        pred = model.predict(X_vec)[0]
                        if hasattr(model, 'predict_proba'):
                            prob = model.predict_proba(X_vec)[0]
                        else:
                            # Для SVM без вероятностей
                            decision_score = model.decision_function(X_vec)
                            # decision_score может быть массивом, берем первый элемент
                            if isinstance(decision_score, np.ndarray) and len(decision_score) == 1:
                                decision_score = decision_score[0]
                            prob = self._decision_to_probability([decision_score])[0]

                    base_results[name] = {
                        'prediction': pred,
                        'probability': prob,
                        'sentiment': "POSITIVE" if pred == self.positive_label else "NEGATIVE"
                    }
                except Exception as e:
                    print(f"⚠️  Ошибка в модели {name}: {e}")
                    continue
        else:
            # Для стекинга - получаем из named_estimators_
            try:
                for name, model in self.stacking_model.named_estimators_.items():
                    try:
                        if name == 'catboost':
                            X_dense = X_vec.toarray()
                            pred = model.predict(X_dense)[0]
                            prob = model.predict_proba(X_dense)[0]
                        else:
                            pred = model.predict(X_vec)[0]
                            if hasattr(model, 'predict_proba'):
                                prob = model.predict_proba(X_vec)[0]
                            else:
                                decision_score = model.decision_function(X_vec)
                                if isinstance(decision_score, np.ndarray) and len(decision_score) == 1:
                                    decision_score = decision_score[0]
                                prob = self._decision_to_probability([decision_score])[0]

                        base_results[name] = {
                            'prediction': pred,
                            'probability': prob,
                            'sentiment': "POSITIVE" if pred == self.positive_label else "NEGATIVE"
                        }
                    except Exception as e:
                        print(f"⚠️  Ошибка в модели {name}: {e}")
                        continue
            except Exception as e:
                print(f"⚠️  Ошибка при получении предсказаний базовых моделей: {e}")

        return base_results

    def _get_consensus(self, base_predictions):
        """
        Анализ консенсуса базовых моделей
        """
        predictions = [data['prediction'] for data in base_predictions.values()]
        positive_votes = sum(1 for p in predictions if p == self.positive_label)
        total_votes = len(predictions)

        return {
            'positive_votes': positive_votes,
            'negative_votes': total_votes - positive_votes,
            'total_votes': total_votes,
            'consensus_ratio': max(positive_votes, total_votes - positive_votes) / total_votes,
            'unanimous': positive_votes == total_votes or positive_votes == 0
        }

    def evaluate(self, test_data):
        """
        Оценка модели на тестовых данных
        """
        X_test, y_test = self.prepare_data(test_data)

        predictions, probabilities = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print("\n📊 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        target_names = [f'NEGATIVE({self.negative_label})', f'POSITIVE({self.positive_label})']
        print(classification_report(y_test, predictions, target_names=target_names))

        # Матрица ошибок
        print("\n📈 МАТРИЦА ОШИБОК:")
        cm = confusion_matrix(y_test, predictions)
        print(f"               Предсказано {self.negative_label}  Предсказано {self.positive_label}")
        print(f"Реально {self.negative_label}:     {cm[0][0]:^10}        {cm[0][1]:^10}")
        print(f"Реально {self.positive_label}:     {cm[1][0]:^10}        {cm[1][1]:^10}")

        return accuracy

    def analyze_model_performance(self, data):
        """
        Детальный анализ производительности всех моделей
        """
        X, y = self.prepare_data(data)
        X_vec = self.vectorizer.transform(X)

        print("\n📊 АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛЕЙ:")
        print("=" * 60)

        base_accuracies = {}

        if self.use_blending:
            # Для блендинга - базовые модели обучены отдельно
            for name, model in self.base_models.items():
                if name == 'catboost':
                    X_dense = X_vec.toarray()
                    pred = model.predict(X_dense)
                else:
                    pred = model.predict(X_vec)

                accuracy = accuracy_score(y, pred)
                base_accuracies[name] = accuracy
                print(f"   {name.upper():<12}: {accuracy:.3f}")
        else:
            # Для стекинга - получаем предсказания через кросс-валидацию или используем финальные estimators
            try:
                # Получаем обученные базовые модели из stacking classifier
                for name, model in self.stacking_model.named_estimators_.items():
                    if name == 'catboost':
                        X_dense = X_vec.toarray()
                        pred = model.predict(X_dense)
                    else:
                        pred = model.predict(X_vec)

                    accuracy = accuracy_score(y, pred)
                    base_accuracies[name] = accuracy
                    print(f"   {name.upper():<12}: {accuracy:.3f}")
            except Exception as e:
                print(f"   ⚠️  Не удалось получить предсказания базовых моделей: {e}")
                # Альтернатива: используем основные метрики из обучения
                return {'ensemble_accuracy': None, 'improvement': None}

        # Оценка ансамбля
        ensemble_pred, _ = self.predict(X)
        ensemble_accuracy = accuracy_score(y, ensemble_pred)
        print(f"   {'ENSEMBLE':<12}: {ensemble_accuracy:.3f}")

        # Улучшение по сравнению с лучшей базовой моделью
        if base_accuracies:
            best_base_accuracy = max(base_accuracies.values())
            improvement = ensemble_accuracy - best_base_accuracy
            print(f"\n   📈 Улучшение над лучшей моделью: {improvement:.3f}")
            print(f"   📈 Относительное улучшение: {improvement / best_base_accuracy * 100:.1f}%")
        else:
            improvement = 0
            print(f"\n   ⚠️  Не удалось вычислить улучшение")

        return {
            'base_accuracies': base_accuracies,
            'ensemble_accuracy': ensemble_accuracy,
            'improvement': improvement
        }

    def save_model(self, filename):
        """
        Сохранение модели
        """
        if self.use_blending:
            to_save = {
                'base_models': self.base_models,
                'meta_model': self.meta_model,
                'vectorizer': self.vectorizer,
                'use_blending': self.use_blending,
                'meta_model_type': self.meta_model_type,
                'positive_label': self.positive_label,
                'negative_label': self.negative_label
            }
        else:
            to_save = {
                'stacking_model': self.stacking_model,
                'vectorizer': self.vectorizer,
                'use_blending': self.use_blending,
                'meta_model_type': self.meta_model_type,
                'positive_label': self.positive_label,
                'negative_label': self.negative_label
            }

        joblib.dump(to_save, filename)
        print(f"💾 Модель сохранена: {filename}")

    def load_model(self, filename):
        """
        Загрузка модели
        """
        loaded = joblib.load(filename)

        self.vectorizer = loaded['vectorizer']
        self.use_blending = loaded['use_blending']
        self.meta_model_type = loaded['meta_model_type']
        self.positive_label = loaded.get('positive_label', 1)
        self.negative_label = loaded.get('negative_label', 0)

        if self.use_blending:
            self.base_models = loaded['base_models']
            self.meta_model = loaded['meta_model']
        else:
            self.stacking_model = loaded['stacking_model']

        self.is_trained = True
        print(f"📥 Модель загружена: {filename}")


# Сравнение разных стратегий ансамблирования
def compare_ensemble_strategies(train_data, val_data, test_data):
    """
    Сравнение стекинга и блендинга с разными мета-моделями
    """
    print("🔬 СРАВНЕНИЕ СТРАТЕГИЙ АНСАМБЛИРОВАНИЯ")
    print("=" * 60)

    strategies = [
        ('blending_logistic', True, 'logistic'),
        ('blending_svm', True, 'svm'),
        ('blending_rf', True, 'random_forest'),
        ('stacking_logistic', False, 'logistic'),
        ('stacking_svm', False, 'svm'),  # Теперь будет использовать SVC с probability=True
    ]

    results = {}

    for name, use_blending, meta_model in strategies:
        print(f"\n🎯 {name.upper()}:")
        ensemble = StackingSentimentClassifier(
            use_blending=use_blending,
            meta_model=meta_model
        )

        if use_blending:
            ensemble.train(train_data, val_data)
        else:
            ensemble.train(train_data, val_data)

        test_accuracy = ensemble.evaluate(test_data)
        results[name] = {
            'model': ensemble,
            'accuracy': test_accuracy
        }

    # Сравнение результатов
    print("\n📊 ИТОГОВОЕ СРАВНЕНИЕ:")
    print("=" * 40)
    for name, result in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"   {name:<20}: {result['accuracy']:.3f}")

    return results


# Пример использования
def main():
    """
    Пример использования стекинг/блендинг классификатора
    """
    train_data = read_jsonl_basic('../../util/news_sentiment_train.jsonl')
    val_data = read_jsonl_basic('../../util/news_sentiment_val.jsonl')
    test_data = read_jsonl_basic('../../util/news_sentiment_test.jsonl')

    print(f"📊 Данные: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Тестирование блендинга
    print("\n" + "=" * 50)
    print("🎯 ТЕСТИРОВАНИЕ БЛЕНДИНГА")
    blending_classifier = StackingSentimentClassifier(
        use_blending=True,
        meta_model='logistic'
    )

    blending_results = blending_classifier.train(train_data, val_data)

    blending_classifier.analyze_model_performance(test_data)

    print("\n" + "=" * 50)
    print("🎯 ТЕСТИРОВАНИЕ СТЕКИНГА")
    stacking_classifier = StackingSentimentClassifier(
        use_blending=False,
        meta_model='logistic'
    )

    stacking_classifier.train(train_data, val_data)
    stacking_classifier.analyze_model_performance(test_data)

    blending_classifier.save_model("blending_classifier.pkl")
    stacking_classifier.save_model("stacking_classifier.pkl")

    print("\n" + "=" * 50)
    results = compare_ensemble_strategies(train_data, val_data, test_data)


if __name__ == "__main__":
    main()