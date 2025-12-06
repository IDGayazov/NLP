from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
from scipy.stats import randint, uniform
import warnings
import json

warnings.filterwarnings('ignore')


class SimpleAutoMLMultiClassClassifier:
    """
    Упрощенный AutoML классификатор для многоклассовой классификации
    """

    def __init__(self, max_training_time=300, n_iter=50, random_state=42):
        """
        Args:
            max_training_time: максимальное время обучения (в секундах)
            n_iter: количество итераций случайного поиска
            random_state: для воспроизводимости
        """
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words=None
        )

        # Для кодирования меток
        self.label_encoder = LabelEncoder()

        # Определяем модели и пространства параметров для поиска
        self.models = {
            'logistic': {
                'model': LogisticRegression(random_state=random_state, multi_class='ovr'),
                'params': {
                    'C': uniform(0.001, 100),
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [1000, 2000]
                }
            },
            'svm': {
                'model': SVC(random_state=random_state, probability=True, decision_function_shape='ovr'),
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
        self.random_state = random_state
        self.is_trained = False
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.class_names = None
        self.n_classes = None

        print(f"🚀 Multi-class AutoML инициализирован:")
        print(f"   Максимальное время: {max_training_time} сек")
        print(f"   Итераций поиска: {n_iter}")
        print(f"   Модели для тестирования: {list(self.models.keys())}")

    def prepare_data(self, data):
        """
        Подготовка данных: извлекаем тексты и метки
        """
        texts = [item['text'] for item in data]
        labels = [item['category'] for item in data]
        return texts, labels

    def train(self, train_data, val_data=None):
        """
        Обучение AutoML классификатора для многоклассовой классификации
        """
        print("🎯 АВТОМАТИЗИРОВАННЫЙ ПОДБОР МОДЕЛЕЙ (многоклассовый)...")

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
                    n_iter=self.n_iter // len(self.models),
                    cv=3,
                    scoring='accuracy',
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=0
                )

                search.fit(X_train_vec, y_train_encoded)

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
            print(f"   {model_name:<15}: {score:.3f}")

    def predict(self, texts):
        """
        Предсказание для списка текстов
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)
        predictions_encoded = self.best_model.predict(X_vec)

        # Декодируем предсказания
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        probabilities = self.best_model.predict_proba(X_vec)

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
            'model_type': type(self.best_model).__name__,
            'model_name': self.best_model_name
        }

    def evaluate(self, test_data):
        """
        Оценка модели на тестовых данных для многоклассовой классификации
        """
        X_test, y_test = self.prepare_data(test_data)
        X_test_vec = self.vectorizer.transform(X_test)

        # Кодируем тестовые метки для сравнения
        y_test_encoded = self.label_encoder.transform(y_test)

        y_pred_encoded = self.best_model.predict(X_test_vec)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)

        print("\n📊 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ (Многоклассовая классификация):")
        print(classification_report(y_test, y_pred, digits=4))

        # Матрица ошибок
        print("\n📈 МАТРИЦА ОШИБОК:")
        cm = confusion_matrix(y_test_encoded, y_pred_encoded)

        # Красивый вывод матрицы ошибок
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

        # Заголовок
        header = " " * 15 + "Предсказано →"
        print(header)

        # Имена классов для предсказаний
        pred_header = " " * 10
        for name in class_names:
            pred_header += f"{name[:8]:^8} "
        print(pred_header)

        # Разделитель
        separator = " " * 10 + "─" * (n_classes * 9)
        print(separator)

        # Строки матрицы
        for i, true_name in enumerate(class_names):
            row = f"Истинно {true_name[:8]:<8}│"
            for j in range(n_classes):
                row += f"{cm[i][j]:^8} "
            print(row)

        # Вычисляем диагональ (правильные предсказания)
        diagonal = cm.diagonal()
        total = cm.sum()
        correct = diagonal.sum()

        print(f"\n📊 Статистика:")
        print(f"   Правильно классифицировано: {correct}/{total} ({correct / total:.1%})")

        # Точность по классам
        print(f"\n📊 Accuracy по классам:")
        for i, class_name in enumerate(class_names):
            class_total = cm[i, :].sum()
            if class_total > 0:
                class_correct = cm[i, i]
                print(f"   {class_name}: {class_correct}/{class_total} ({class_correct / class_total:.1%})")

    def get_model_info(self):
        """
        Возвращает информацию о лучшей модели
        """
        return {
            'model_name': self.best_model_name,
            'model_type': type(self.best_model).__name__,
            'best_score': self.best_score,
            'parameters': self.best_params,
            'feature_count': len(self.vectorizer.get_feature_names_out()),
            'n_classes': self.n_classes,
            'class_names': list(self.class_names)
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
            'best_model': self.best_model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'class_names': self.class_names,
            'n_classes': self.n_classes
        }, filename)
        print(f"💾 Multi-class AutoML модель сохранена: {filename}")

    def load_model(self, filename):
        """
        Загрузка модели
        """
        loaded = joblib.load(filename)
        self.best_model = loaded['best_model']
        self.vectorizer = loaded['vectorizer']
        self.label_encoder = loaded['label_encoder']
        self.best_model_name = loaded['best_model_name']
        self.best_score = loaded.get('best_score', 0)
        self.best_params = loaded.get('best_params', {})
        self.class_names = loaded.get('class_names', [])
        self.n_classes = loaded.get('n_classes', 0)
        self.is_trained = True
        print(f"📥 Multi-class AutoML модель загружена: {filename}")


# СПЕЦИАЛИЗИРОВАННЫЙ КЛАССИФИКАТОР ДЛЯ ТОНАЛЬНОСТИ (3 КЛАССА)
class SentimentAutoMLClassifier(SimpleAutoMLMultiClassClassifier):
    """
    Специализированный AutoML классификатор для тональности (негатив, нейтрал, позитив)
    """

    def __init__(self, max_training_time=300, n_iter=50, random_state=42):
        super().__init__(max_training_time, n_iter, random_state)

        # Добавляем специфичные для тональности модели
        self.models.update({
            'svm_rbf': {
                'model': SVC(kernel='rbf', probability=True, random_state=random_state),
                'params': {
                    'C': uniform(0.1, 10),
                    'gamma': uniform(0.001, 0.1)
                }
            },
            'xgboost': {
                'model': None,  # Будет загружена при наличии
                'params': {}
            }
        })

        # Пытаемся добавить XGBoost если установлен
        try:
            from xgboost import XGBClassifier
            self.models['xgboost']['model'] = XGBClassifier(
                random_state=random_state,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            self.models['xgboost']['params'] = {
                'n_estimators': randint(50, 200),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.5, 0.5)
            }
        except ImportError:
            print("XGBoost не установлен, пропускаем...")
            del self.models['xgboost']

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

        return result


def main():
    """
    Пример использования многоклассового AutoML классификатора
    """

    # Функция для загрузки данных (адаптируйте под ваш формат)
    def load_jsonl(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    print("📂 ЗАГРУЗКА ДАННЫХ...")

    # Пример данных для многоклассовой классификации
    # Ваши данные должны иметь поле 'label' вместо 'sentiment'
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

    # Создаем и обучаем классификатор
    print("\n" + "=" * 60)
    print("🎯 ОБУЧЕНИЕ МНОГОКЛАССОВОГО AUTOML")
    print("=" * 60)

    automl_classifier = SimpleAutoMLMultiClassClassifier(
        n_iter=30,
        max_training_time=120  # 2 минуты
    )

    # Показываем распределение классов
    print("\n📊 РАСПРЕДЕЛЕНИЕ КЛАССОВ В ОБУЧАЮЩИХ ДАННЫХ:")
    class_dist = automl_classifier.get_class_distribution(train_subset)
    for class_name, count in class_dist.items():
        print(f"   {class_name}: {count} примеров ({count / len(train_subset):.1%})")

    # Обучаем
    automl_classifier.train(train_subset, val_subset)

    # Оцениваем
    print("\n🧪 ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ...")
    accuracy = automl_classifier.evaluate(test_data)

    # Сохраняем модель
    automl_classifier.save_model('multiclass_automl_model.pkl')

    # Загружаем модель и тестируем
    print("\n🧪 ТЕСТ ЗАГРУЗКИ МОДЕЛИ...")
    loaded_classifier = SimpleAutoMLMultiClassClassifier()
    loaded_classifier.load_model('multiclass_automl_model.pkl')

    # Быстрый тест загруженной модели
    test_text = "Довольно неплохо, но есть небольшие замечания"
    result = loaded_classifier.predict_single(test_text)
    print(f"\n📝 Тест загруженной модели:")
    print(f"   Текст: {test_text}")
    print(f"   Предсказание: {result['prediction']}")
    print(f"   Уверенность: {result['confidence']:.3f}")

    return automl_classifier


def analyze_multiclass_performance(automl_classifier, test_data):
    """
    Детальный анализ производительности для многоклассовой классификации
    """
    from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score

    X_test, y_test = automl_classifier.prepare_data(test_data)
    y_pred, _ = automl_classifier.predict(X_test)

    # Дополнительные метрики
    y_test_encoded = automl_classifier.label_encoder.transform(y_test)
    y_pred_encoded = automl_classifier.label_encoder.transform(y_pred)

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_test_encoded, y_pred_encoded)

    # Precision, Recall, F1 для каждого класса
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_encoded, y_pred_encoded, labels=range(automl_classifier.n_classes)
    )

    print("\n📊 ДЕТАЛЬНЫЙ АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ:")
    print("=" * 60)

    print(f"\n📈 Коэффициент Каппа (Cohen's Kappa): {kappa:.4f}")
    print("   (>0.8: отличное согласие, >0.6: хорошее, >0.4: умеренное)")

    print(f"\n📊 МЕТРИКИ ПО КЛАССАМ:")
    print("-" * 40)
    print(f"{'Класс':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 40)

    for i, class_name in enumerate(automl_classifier.class_names):
        print(f"{class_name:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")

    # Матрица ошибок в виде DataFrame для лучшей визуализации
    cm = confusion_matrix(y_test_encoded, y_pred_encoded)
    cm_df = pd.DataFrame(cm,
                         index=automl_classifier.class_names,
                         columns=automl_classifier.class_names)

    print(f"\n📊 МАТРИЦА ОШИБОК (DataFrame):")
    print(cm_df)

    return {
        'kappa': kappa,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm_df
    }


if __name__ == "__main__":
    print("🚀 ЗАПУСК МНОГОКЛАССОВОГО AUTOML КЛАССИФИКАТОРА")
    print("=" * 80)

    # Запускаем основной пример
    classifier = main()