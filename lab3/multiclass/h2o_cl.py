import h2o
from h2o.automl import H2OAutoML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import joblib
import json
import warnings

warnings.filterwarnings('ignore')


class H2OMultiClassClassifier:
    """
    Многоклассовый классификатор на основе H2O.ai AutoML
    """

    def __init__(self, max_runtime_secs=300, max_models=10, nfolds=5):
        """
        Args:
            max_runtime_secs: максимальное время работы AutoML в секундах
            max_models: максимальное количество моделей для построения
            nfolds: количество фолдов для кросс-валидации
        """
        # Инициализация H2O кластера
        h2o.init()
        print(f"🚀 H2O кластер инициализирован")

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words=None
        )

        # Для кодирования меток
        self.label_encoder = LabelEncoder()

        # H2O AutoML для многоклассовой классификации
        self.aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            max_models=max_models,
            nfolds=nfolds,
            seed=42,
            sort_metric="logloss",  # Для многоклассовой лучше использовать logloss
            verbosity="info"
        )

        self.max_runtime_secs = max_runtime_secs
        self.max_models = max_models
        self.is_trained = False
        self.leader_model = None
        self.class_names = None
        self.n_classes = None

        print(f"🎯 H2O.AI AUTOML ДЛЯ МНОГОКЛАССОВОЙ КЛАССИФИКАЦИИ:")
        print(f"   Время: {max_runtime_secs} сек")
        print(f"   Макс. моделей: {max_models}")
        print(f"   CV фолдов: {nfolds}")
        print(f"   H2O версия: {h2o.__version__}")

    def prepare_features(self, data):
        """
        Подготовка признаков из данных
        """
        texts = [item['text'] for item in data]
        return texts

    def prepare_labels(self, data):
        """
        Подготовка меток из данных
        """
        labels = [item['category'] for item in data]
        return labels

    def _create_h2o_frame(self, texts, labels=None):
        """
        Создает H2O Frame из текстов и меток
        """
        # Векторизация текстов
        X_vec = self.vectorizer.transform(texts)
        X_dense = X_vec.toarray()

        # Создаем DataFrame
        feature_names = [f"feature_{i}" for i in range(X_dense.shape[1])]
        df = pd.DataFrame(X_dense, columns=feature_names)

        if labels is not None:
            df['label'] = labels

        # Конвертируем в H2O Frame
        h2o_frame = h2o.H2OFrame(df)

        if labels is not None:
            # Указываем что label - категориальная переменная
            h2o_frame['label'] = h2o_frame['label'].asfactor()

        return h2o_frame

    def train(self, train_data, val_data=None):
        """
        Обучение H2O AutoML для многоклассовой классификации
        """
        print("🎯 АВТОМАТИЗИРОВАННЫЙ ПОИСК МОДЕЛЕЙ С H2O.AI...")

        # Подготовка данных
        X_train_texts = self.prepare_features(train_data)
        y_train = self.prepare_labels(train_data)

        # Кодируем метки
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.class_names = self.label_encoder.classes_
        self.n_classes = len(self.class_names)

        # Проверяем классы
        print(f"📊 Обнаружено {self.n_classes} классов: {list(self.class_names)}")
        print(f"📊 Размер данных: {len(X_train_texts)} примеров")

        # Показываем распределение классов
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            percentage = (count / len(y_train)) * 100
            print(f"   {cls}: {count} примеров ({percentage:.1f}%)")

        # Создаем векторйзер и преобразуем данные
        print("📊 Векторизация текстов...")
        self.vectorizer.fit(X_train_texts)

        # Создаем H2O Frame для обучения
        print("📊 Создание H2O Frame...")
        train_frame = self._create_h2o_frame(X_train_texts, y_train)

        print(f"   Размерность: {train_frame.shape}")
        print(f"   Колонки: {train_frame.columns[:5]}...")  # Показываем только первые 5

        # Определяем features и target
        x = train_frame.columns[:-1]  # Все кроме последней колонки (target)
        y = 'label'

        print(f"   Features: {len(x)} признаков")
        print(f"   Target: {y}")
        print(f"   Классы: {self.n_classes}")

        # Запуск AutoML
        print(f"\n🤖 ЗАПУСК H2O AUTOML...")
        print(f"   Это займет примерно {self.max_runtime_secs} секунд")

        self.aml.train(x=x, y=y, training_frame=train_frame)
        self.is_trained = True

        # Получаем лидер-модель
        self.leader_model = self.aml.leader

        # Показываем результаты
        self._show_training_summary()

        # Валидация если есть данные
        if val_data:
            print("\n🔍 ОЦЕНКА НА ВАЛИДАЦИИ:")
            self.evaluate(val_data)

    def _show_training_summary(self):
        """
        Показывает результаты обучения AutoML
        """
        print("\n📊 РЕЗУЛЬТАТЫ H2O AUTOML:")
        print("=" * 60)

        # Лидерборд
        lb = self.aml.leaderboard
        print("🏆 ЛИДЕРБОРД МОДЕЛЕЙ:")
        print(lb.head(rows=min(10, len(lb))))  # Показываем до 10 моделей

        # Лучшая модель
        print(f"\n🎯 ЛУЧШАЯ МОДЕЛЬ:")
        print(f"   Алгоритм: {self.leader_model.algo}")
        print(f"   Model ID: {self.leader_model.model_id}")
        print(f"   Количество классов: {self.n_classes}")

        # Метрики лучшей модели
        try:
            performance = self.leader_model.model_performance()

            # Для многоклассовой классификации используем другие метрики
            if hasattr(performance, 'logloss'):
                print(f"   Logloss: {performance.logloss():.4f}")
            if hasattr(performance, 'mean_per_class_error'):
                print(f"   Mean per class error: {performance.mean_per_class_error():.4f}")
            if hasattr(performance, 'mse'):
                print(f"   MSE: {performance.mse():.4f}")
            if hasattr(performance, 'accuracy'):
                print(f"   Accuracy: {performance.accuracy():.4f}")

            # Конфузионная матрица для лучшей модели (если доступно)
            try:
                cm = performance.confusion_matrix()
                if cm is not None:
                    print(f"\n📊 Конфузионная матрица (первые 5x5):")
                    cm_df = cm.as_data_frame()
                    print(cm_df.head())
            except:
                pass

        except Exception as e:
            print(f"   Метрики: {e}")

    def predict(self, data):
        """
        Предсказание для данных
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        texts = self.prepare_features(data)

        # Создаем H2O Frame для предсказания
        predict_frame = self._create_h2o_frame(texts)

        # Предсказания
        predictions = self.leader_model.predict(predict_frame)

        # Конвертируем в numpy arrays
        pred_array = predictions['predict'].as_data_frame().values.flatten()

        # Получаем вероятности для всех классов
        prob_columns = [f'p{i}' for i in range(self.n_classes)]
        if all(col in predictions.columns for col in prob_columns):
            prob_array = predictions[prob_columns].as_data_frame().values
        else:
            # Если столбцы называются по-другому
            prob_columns = [col for col in predictions.columns if col.startswith('p')]
            prob_array = predictions[prob_columns].as_data_frame().values

        return pred_array, prob_array

    def predict_single(self, text):
        """
        Предсказание для одного текста
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        # Создаем H2O Frame для одного текста
        predict_frame = self._create_h2o_frame([text])

        # Предсказание
        prediction = self.leader_model.predict(predict_frame)

        # Извлекаем результаты
        pred = prediction['predict'].as_data_frame().values[0][0]

        # Получаем вероятности для всех классов
        prob_columns = [f'p{i}' for i in range(self.n_classes)]
        if all(col in prediction.columns for col in prob_columns):
            probabilities = prediction[prob_columns].as_data_frame().values[0]
        else:
            # Если столбцы называются по-другому
            prob_columns = sorted([col for col in prediction.columns if col.startswith('p')])
            probabilities = prediction[prob_columns].as_data_frame().values[0]

        # Создаем словарь вероятностей по классам
        class_probabilities = {}
        for i, class_name in enumerate(self.class_names):
            if i < len(probabilities):
                class_probabilities[class_name] = probabilities[i]
            else:
                class_probabilities[class_name] = 0.0

        # Находим индекс предсказанного класса
        pred_idx = list(self.class_names).index(pred) if pred in self.class_names else 0

        # Сортируем вероятности
        sorted_probs = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_probs[:3]

        result = {
            'prediction': pred,
            'confidence': f"{probabilities[pred_idx]:.3f}",
            'probabilities': class_probabilities,
            'top_3_predictions': top_3,
            'model_type': self.leader_model.algo
        }

        return result

    def evaluate(self, test_data):
        """
        Оценка на тестовых данных для многоклассовой классификации
        """
        X_test_texts = self.prepare_features(test_data)
        y_test = self.prepare_labels(test_data)

        # Предсказания
        predictions, probabilities = self.predict(test_data)

        # Точность
        accuracy = accuracy_score(y_test, predictions)

        print(f"📊 ТЕСТОВАЯ ТОЧНОСТЬ: {accuracy:.4f}")

        # Детальный отчет
        print("\n📈 ДЕТАЛЬНЫЙ ОТЧЕТ:")
        print(classification_report(y_test, predictions, digits=4))

        # Матрица ошибок
        print("\n📊 МАТРИЦА ОШИБОК:")
        y_test_encoded = self.label_encoder.transform(y_test)
        predictions_encoded = self.label_encoder.transform(predictions)
        cm = confusion_matrix(y_test_encoded, predictions_encoded)

        # Выводим матрицу ошибок
        self._print_confusion_matrix(cm, self.class_names)

        # Дополнительные метрики
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision_macro = precision_score(y_test_encoded, predictions_encoded, average='macro')
        recall_macro = recall_score(y_test_encoded, predictions_encoded, average='macro')
        f1_macro = f1_score(y_test_encoded, predictions_encoded, average='macro')

        precision_weighted = precision_score(y_test_encoded, predictions_encoded, average='weighted')
        recall_weighted = recall_score(y_test_encoded, predictions_encoded, average='weighted')
        f1_weighted = f1_score(y_test_encoded, predictions_encoded, average='weighted')

        print(f"\n📊 СРЕДНИЕ МЕТРИКИ:")
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

        # Упрощенный вывод для большого количества классов
        if n_classes > 8:
            print("   (Матрица слишком большая для отображения)")
            print(f"   Правильно классифицировано: {np.trace(cm)}/{np.sum(cm)} ({np.trace(cm) / np.sum(cm):.1%})")
            return

        # Заголовок
        header = " " * 10 + "Предсказано →"
        print(header)

        # Имена классов для предсказаний
        pred_header = " " * 5
        for name in class_names[:n_classes]:
            pred_header += f"{str(name)[:6]:^6} "
        print(pred_header)

        # Разделитель
        separator = " " * 5 + "─" * (n_classes * 7 + 1)
        print(separator)

        # Строки матрицы
        for i, true_name in enumerate(class_names[:n_classes]):
            row = f"{str(true_name)[:5]:>5} │"
            for j in range(n_classes):
                row += f"{cm[i][j]:^6} "
            print(row)

        # Статистика
        diagonal = cm.diagonal()
        total = cm.sum()
        correct = diagonal.sum()

        print(f"\n📊 Статистика:")
        print(f"   Правильно классифицировано: {correct}/{total} ({correct / total:.1%})")

    def get_model_info(self):
        """
        Возвращает информацию о моделях
        """
        if not self.is_trained:
            return {"error": "Модель не обучена"}

        lb = self.aml.leaderboard.as_data_frame()

        # Выбираем только нужные колонки, если они существуют
        available_columns = []
        for col in ['model_id', 'auc', 'logloss', 'mean_per_class_error', 'mse', 'rmse']:
            if col in lb.columns:
                available_columns.append(col)

        if available_columns:
            top_models = lb.head(5)[available_columns].to_dict('records')
        else:
            top_models = lb.head(5).to_dict('records')

        return {
            "leader_model": self.leader_model.model_id,
            "leader_algorithm": self.leader_model.algo,
            "n_classes": self.n_classes,
            "class_names": list(self.class_names),
            "top_models": top_models,
            "total_models_trained": len(lb),
            "feature_count": len(self.vectorizer.get_feature_names_out())
        }

    def get_class_distribution(self, data):
        """
        Возвращает распределение классов в данных
        """
        labels = self.prepare_labels(data)
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))

    def save_model(self, filename):
        """
        Сохранение модели
        """
        if not self.is_trained:
            print("❌ Нельзя сохранить необученную модель")
            return

        # Сохраняем H2O модель
        model_path = h2o.save_model(model=self.leader_model, path=filename, force=True)

        # Сохраняем векторйзер и метаданные
        joblib.dump({
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'model_path': model_path,
            'model_id': self.leader_model.model_id,
            'class_names': self.class_names,
            'n_classes': self.n_classes
        }, f"{filename}_meta.pkl")

        print(f"💾 Модель сохранена: {model_path}")

    def load_model(self, filename):
        """
        Загрузка модели
        """
        # Загружаем метаданные
        meta = joblib.load(f"{filename}_meta.pkl")
        self.vectorizer = meta['vectorizer']
        self.label_encoder = meta['label_encoder']
        self.class_names = meta['class_names']
        self.n_classes = meta['n_classes']

        # Загружаем H2O модель
        self.leader_model = h2o.load_model(meta['model_path'])
        self.is_trained = True

        print(f"📥 Модель загружена: {meta['model_id']}")
        print(f"📥 Количество классов: {self.n_classes}")
        print(f"📥 Имена классов: {list(self.class_names)}")

    def __del__(self):
        """
        Закрытие H2O кластера при удалении объекта
        """
        try:
            h2o.cluster().shutdown()
        except:
            pass


# СПЕЦИАЛИЗИРОВАННЫЙ КЛАССИФИКАТОР ДЛЯ ТОНАЛЬНОСТИ (3 КЛАССА)
class H2OSentimentClassifier(H2OMultiClassClassifier):
    """
    Специализированный H2O классификатор для тональности (негатив, нейтрал, позитив)
    """

    def __init__(self, max_runtime_secs=300, max_models=10, nfolds=5):
        super().__init__(
            max_runtime_secs=max_runtime_secs,
            max_models=max_models,
            nfolds=nfolds
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

        return result


# Упрощенная версия для быстрого старта
class SimpleH2OMultiClassClassifier:
    """
    Упрощенная версия H2O классификатора для многоклассовой классификации
    """

    def __init__(self, time_seconds=120):
        h2o.init()
        print(f"🚀 H2O кластер инициализирован")

        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.label_encoder = LabelEncoder()
        self.aml = H2OAutoML(max_runtime_secs=time_seconds, seed=42, verbosity="info")
        self.is_trained = False
        self.class_names = None

        print(f"⚡ SimpleH2O (многоклассовый): {time_seconds} сек")

    def train_and_test(self, train_data, test_data):
        """Обучает и тестирует за один шаг"""
        # Подготовка данных
        train_texts = [item['text'] for item in train_data]
        train_labels = [item['category'] for item in train_data]
        test_texts = [item['text'] for item in test_data]
        test_labels = [item['category'] for item in test_data]

        # Кодируем метки
        train_labels_encoded = self.label_encoder.fit_transform(train_labels)
        test_labels_encoded = self.label_encoder.transform(test_labels)
        self.class_names = self.label_encoder.classes_

        print(f"📚 Обучение на {len(train_texts)} примерах...")
        print(f"📊 Количество классов: {len(self.class_names)}")
        print(f"📊 Классы: {list(self.class_names)}")

        # Векторизация и создание H2O Frame
        self.vectorizer.fit(train_texts)

        train_features = self.vectorizer.transform(train_texts).toarray()
        test_features = self.vectorizer.transform(test_texts).toarray()

        # Создаем DataFrame
        feature_names = [f"f_{i}" for i in range(train_features.shape[1])]

        train_df = pd.DataFrame(train_features, columns=feature_names)
        train_df['target'] = train_labels  # Используем оригинальные метки

        test_df = pd.DataFrame(test_features, columns=feature_names)
        test_df['target'] = test_labels

        # H2O Frames
        train_frame = h2o.H2OFrame(train_df)
        test_frame = h2o.H2OFrame(test_df)
        train_frame['target'] = train_frame['target'].asfactor()
        test_frame['target'] = test_frame['target'].asfactor()

        # AutoML
        print("🤖 Запуск AutoML...")
        self.aml.train(x=feature_names, y='target', training_frame=train_frame)
        self.is_trained = True

        # Предсказания
        predictions = self.aml.leader.predict(test_frame)
        pred_array = predictions['predict'].as_data_frame().values.flatten()

        accuracy = accuracy_score(test_labels, pred_array)

        print(f"✅ Результаты:")
        print(f"   Лучшая модель: {self.aml.leader.algo}")
        print(f"   Точность: {accuracy:.4f}")

        # Classification report
        print("\n📈 Детальный отчет:")
        print(classification_report(test_labels, pred_array, digits=3))

        return accuracy

    def __del__(self):
        """Закрытие H2O кластера"""
        try:
            h2o.cluster().shutdown()
        except:
            pass


# ПРИМЕР ИСПОЛЬЗОВАНИЯ
def main():
    """
    Основной пример использования H2O AutoML для многоклассовой классификации
    """

    # Функция для загрузки данных
    def load_jsonl(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    print("=" * 80)
    print("🚀 H2O.AI AUTOML ДЛЯ МНОГОКЛАССОВОЙ КЛАССИФИКАЦИИ")
    print("=" * 80)

    train_data = load_jsonl('../util/news_category_train.jsonl')
    test_data = load_jsonl('../util/news_category_test.jsonl')

    print(f"📁 Train: {len(train_data)} примеров")
    print(f"📁 Test: {len(test_data)} примеров")

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

    # Вариант 1: Полная версия H2O
    print("\n" + "=" * 50)
    print("🎯 ВАРИАНТ 1: ПОЛНАЯ ВЕРСИЯ H2O AUTOML")

    try:
        h2o_model = H2OMultiClassClassifier(max_runtime_secs=180)  # 3 минуты

        # Показываем распределение классов
        print("\n📊 РАСПРЕДЕЛЕНИЕ КЛАССОВ:")
        class_dist = h2o_model.get_class_distribution(train_subset)
        for class_name, count in class_dist.items():
            percentage = (count / len(train_subset)) * 100
            print(f"   {class_name}: {count} примеров ({percentage:.1f}%)")

        h2o_model.train(train_subset, val_subset)
        h2o_accuracy = h2o_model.evaluate(test_data)

        # Информация о моделях
        model_info = h2o_model.get_model_info()
        print(f"\n📋 ИНФОРМАЦИЯ О МОДЕЛЯХ:")
        print(f"   Лучшая модель: {model_info['leader_algorithm']}")
        print(f"   Количество классов: {model_info['n_classes']}")
        print(f"   Всего моделей: {model_info['total_models_trained']}")
        print(f"   Классы: {model_info['class_names']}")

        # Сохранение модели
        h2o_model.save_model("h2o_multiclass_model")

    except Exception as e:
        print(f"❌ Ошибка H2O: {e}")
        import traceback
        traceback.print_exc()
        h2o_accuracy = 0

    # Вариант 2: Упрощенная версия
    print("\n" + "=" * 50)
    print("🎯 ВАРИАНТ 2: УПРОЩЕННАЯ ВЕРСИЯ")

    try:
        simple_h2o = SimpleH2OMultiClassClassifier(time_seconds=120)  # 2 минуты
        simple_accuracy = simple_h2o.train_and_test(train_subset[:50], test_data[:20])
    except Exception as e:
        print(f"❌ Ошибка упрощенной версии: {e}")
        import traceback
        traceback.print_exc()
        simple_accuracy = 0

    # Завершение работы H2O
    print("\n🔄 Завершение работы H2O кластера...")
    try:
        h2o.cluster().shutdown()
        print("✅ H2O кластер остановлен")
    except:
        print("⚠️  Не удалось остановить H2O кластер")


if __name__ == "__main__":
    print("🚀 ЗАПУСК H2O.AI AUTOML ДЛЯ МНОГОКЛАССОВОЙ КЛАССИФИКАЦИИ")
    print("=" * 80)

    # Запускаем основной пример
    main()