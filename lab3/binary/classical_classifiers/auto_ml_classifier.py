import warnings

import joblib
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from util.jsonl_process import read_jsonl_basic

warnings.filterwarnings('ignore')


class SimpleAutoMLSentimentClassifier:
    """
    Упрощенный AutoML классификатор тональности с RandomizedSearchCV
    """

    def __init__(self, max_training_time=300, n_iter=50,
                 positive_label=1, negative_label=0, random_state=42):
        """
        Args:
            max_training_time: максимальное время обучения (в секундах)
            n_iter: количество итераций случайного поиска
            positive_label: метка положительного класса
            negative_label: метка отрицательного класса
            random_state: для воспроизводимости
        """
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words=None
        )

        # Определяем модели и пространства параметров для поиска
        self.models = {
            'logistic': {
                'model': LogisticRegression(random_state=random_state),
                'params': {
                    'C': uniform(0.001, 100),
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [1000, 2000]
                }
            },
            'svm': {
                'model': SVC(random_state=random_state, probability=True),
                'params': {
                    'C': uniform(0.1, 10),
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto'] + list(uniform(0.001, 0.1).rvs(5))
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=random_state),
                'params': {
                    'n_estimators': randint(50, 300),
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'naive_bayes': {
                'model': MultinomialNB(),
                'params': {
                    'alpha': uniform(0.001, 2.0)
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=random_state),
                'params': {
                    'n_estimators': randint(50, 200),
                    'learning_rate': uniform(0.01, 0.3),
                    'max_depth': randint(3, 10),
                    'min_samples_split': randint(2, 20)
                }
            }
        }

        self.max_training_time = max_training_time
        self.n_iter = n_iter
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.random_state = random_state
        self.is_trained = False
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0

        print(f"🚀 Simple AutoML инициализирован:")
        print(f"   Максимальное время: {max_training_time} сек")
        print(f"   Итераций поиска: {n_iter}")
        print(f"   Модели для тестирования: {list(self.models.keys())}")

    def prepare_data(self, data):
        """
        Подготовка данных: извлекаем тексты и метки
        """
        texts = [item['text'] for item in data]
        labels = [item['sentiment'] for item in data]
        return texts, labels

    def train(self, train_data, val_data=None):
        """
        Обучение AutoML классификатора
        """
        print("🎯 АВТОМАТИЗИРОВАННЫЙ ПОДБОР МОДЕЛЕЙ...")

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
        print(f"   Количество примеров: {len(y_train)}")

        # Автоматический подбор моделей
        print("\n🤖 ЗАПУСК СЛУЧАЙНОГО ПОИСКА ПО МОДЕЛЯМ...")

        best_models = {}

        for model_name, model_config in self.models.items():
            print(f"   🔍 Поиск параметров для {model_name}...")

            try:
                # Случайный поиск по гиперпараметрам
                search = RandomizedSearchCV(
                    model_config['model'],
                    model_config['params'],
                    n_iter=self.n_iter // len(self.models),  # Распределяем итерации
                    cv=3,
                    scoring='accuracy',
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=0
                )

                search.fit(X_train_vec, y_train)

                best_models[model_name] = {
                    'model': search.best_estimator_,
                    'score': search.best_score_,
                    'params': search.best_params_
                }

                print(f"      ✅ Лучшая точность: {search.best_score_:.3f}")

            except Exception as e:
                print(f"      ❌ Ошибка при поиске для {model_name}: {e}")
                continue

        # Выбираем лучшую модель
        if best_models:
            self.best_model_name = max(best_models.keys(), key=lambda x: best_models[x]['score'])
            self.best_model = best_models[self.best_model_name]['model']
            self.best_score = best_models[self.best_model_name]['score']
            self.best_params = best_models[self.best_model_name]['params']

            print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {self.best_model_name}")
            print(f"   Точность: {self.best_score:.3f}")
            print(f"   Параметры: {self.best_params}")

            self.is_trained = True

            # Показываем сравнение всех моделей
            self._show_model_comparison(best_models)

            # Оценка на валидации, если есть
            if val_data:
                val_accuracy = self.evaluate(val_data)
                print(f"✅ Точность на val: {val_accuracy:.3f}")
        else:
            raise Exception("Не удалось обучить ни одну модель!")

    def _show_model_comparison(self, best_models):
        """
        Показывает сравнение всех протестированных моделей
        """
        print(f"\n📊 СРАВНЕНИЕ МОДЕЛЕЙ:")
        print("-" * 50)

        for model_name, results in sorted(best_models.items(),
                                          key=lambda x: x[1]['score'], reverse=True):
            score = results['score']
            params = results['params']
            print(f"   {model_name:<15}: {score:.3f}")

    def predict(self, texts):
        """
        Предсказание для списка текстов
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)
        predictions = self.best_model.predict(X_vec)
        probabilities = self.best_model.predict_proba(X_vec)

        return predictions, probabilities

    def predict_single(self, text):
        """
        Предсказание для одного текста с детальной информацией
        """
        predictions, probabilities = self.predict([text])
        pred = predictions[0]
        prob = probabilities[0]

        # Определяем вероятности для каждого класса
        if self.best_model.classes_[0] == self.positive_label:
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
            'confidence': max(pos_prob, neg_prob),
            'model_type': type(self.best_model).__name__,
            'model_name': self.best_model_name
        }

    def evaluate(self, test_data):
        """
        Оценка модели на тестовых данных
        """
        X_test, y_test = self.prepare_data(test_data)
        X_test_vec = self.vectorizer.transform(X_test)

        y_pred = self.best_model.predict(X_test_vec)
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

    def get_model_info(self):
        """
        Возвращает информацию о лучшей модели
        """
        return {
            'model_name': self.best_model_name,
            'model_type': type(self.best_model).__name__,
            'best_score': self.best_score,
            'parameters': self.best_params,
            'feature_count': len(self.vectorizer.get_feature_names_out())
        }

    def save_model(self, filename):
        """
        Сохранение модели
        """
        joblib.dump({
            'best_model': self.best_model,
            'vectorizer': self.vectorizer,
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'positive_label': self.positive_label,
            'negative_label': self.negative_label
        }, filename)
        print(f"💾 AutoML модель сохранена: {filename}")

    def load_model(self, filename):
        """
        Загрузка модели
        """
        loaded = joblib.load(filename)
        self.best_model = loaded['best_model']
        self.vectorizer = loaded['vectorizer']
        self.best_model_name = loaded['best_model_name']
        self.best_score = loaded.get('best_score', 0)
        self.best_params = loaded.get('best_params', {})
        self.positive_label = loaded.get('positive_label', 1)
        self.negative_label = loaded.get('negative_label', 0)
        self.is_trained = True
        print(f"📥 AutoML модель загружена: {filename}")


# Вариант 2: С использованием TPOT (Tree-based Pipeline Optimization Tool)
try:
    from tpot import TPOTClassifier


    class TPOTSentimentClassifier:
        """
        AutoML классификатор на основе TPOT
        """

        def __init__(self, generations=5, population_size=20,
                     max_time_mins=5, cv=3, random_state=42):
            self.vectorizer = TfidfVectorizer(max_features=5000)
            self.tpot = TPOTClassifier(
                generations=generations,
                population_size=population_size,
                cv=cv,
                random_state=random_state,
                verbosity=2,
                max_time_mins=max_time_mins,
                n_jobs=-1
            )
            self.is_trained = False

        def train(self, train_data, val_data=None):
            texts = [item['text'] for item in train_data]
            labels = [item['sentiment'] for item in train_data]

            X_vec = self.vectorizer.fit_transform(texts)
            self.tpot.fit(X_vec, labels)
            self.is_trained = True

        def predict(self, texts):
            X_vec = self.vectorizer.transform(texts)
            return self.tpot.predict(X_vec), self.tpot.predict_proba(X_vec)

except ImportError:
    print("TPOT не установлен. Установите: pip install tpot")


# Сравнение разных подходов AutoML
def compare_automl_approaches(train_data, val_data, test_data):
    """
    Сравнение разных AutoML подходов
    """
    print("🔬 СРАВНЕНИЕ AUTOML ПОДХОДОВ")
    print("=" * 50)

    approaches = {}

    # Simple AutoML
    print("\n🎯 SIMPLE AUTOML:")
    simple_automl = SimpleAutoMLSentimentClassifier(n_iter=30, max_training_time=180)
    simple_automl.train(train_data, val_data)
    test_accuracy = simple_automl.evaluate(test_data)
    approaches['simple_automl'] = {
        'classifier': simple_automl,
        'accuracy': test_accuracy
    }

    # TPOT (если доступен)
    try:
        from tpot import TPOTClassifier
        print("\n🎯 TPOT:")
        tpot_classifier = TPOTSentimentClassifier(max_time_mins=2)  # Быстрый тест
        tpot_classifier.train(train_data, val_data)
        tpot_accuracy = tpot_classifier.evaluate(test_data)
        approaches['tpot'] = {
            'classifier': tpot_classifier,
            'accuracy': tpot_accuracy
        }
    except ImportError:
        print("   TPOT не установлен, пропускаем...")

    # Ручная настройка для сравнения
    print("\n🎯 РУЧНАЯ НАСТРОЙКА (LogisticRegression):")
    from sklearn.linear_model import LogisticRegression
    manual_classifier = SimpleAutoMLSentimentClassifier(n_iter=1, max_training_time=10)
    manual_classifier.models = {
        'logistic': {
            'model': LogisticRegression(random_state=42),
            'params': {'C': [1.0], 'max_iter': [1000]}
        }
    }
    manual_classifier.train(train_data, val_data)
    manual_accuracy = manual_classifier.evaluate(test_data)
    approaches['manual'] = {
        'classifier': manual_classifier,
        'accuracy': manual_accuracy
    }

    # Сравнение результатов
    print("\n📊 ИТОГОВОЕ СРАВНЕНИЕ:")
    print("=" * 30)
    for name, result in sorted(approaches.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        model_info = result['classifier'].get_model_info()
        print(f"   {name:<15}: {result['accuracy']:.3f} (модель: {model_info['model_name']})")

    return approaches

def main():
    """
    Пример использования AutoML классификатора
    """
    train_data = read_jsonl_basic('../../util/news_sentiment_train.jsonl')
    val_data = read_jsonl_basic('../../util/news_sentiment_val.jsonl')
    test_data = read_jsonl_basic('../../util/news_sentiment_test.jsonl')

    print(f"📊 Данные: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Тестирование Simple AutoML
    print("\n" + "=" * 50)
    automl_classifier = SimpleAutoMLSentimentClassifier(
        n_iter=50,
        max_training_time=300  # 5 минут
    )

    automl_classifier.train(train_data, val_data)

    # Сохранение модели
    automl_classifier.save_model("simple_automl_classifier.pkl")

    # Сравнение подходов
    print("\n" + "=" * 50)
    approaches = compare_automl_approaches(train_data, val_data, test_data)


if __name__ == "__main__":
    main()