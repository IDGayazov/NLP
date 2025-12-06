import h2o
from h2o.automl import H2OAutoML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore')


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


class H2OSentimentClassifier:
    """
    Классификатор тональности на основе H2O.ai AutoML
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

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words=None
        )

        self.aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            max_models=max_models,
            nfolds=nfolds,
            seed=42,
            sort_metric="AUC"
        )

        self.max_runtime_secs = max_runtime_secs
        self.max_models = max_models
        self.is_trained = False
        self.leader_model = None

        print(f"🚀 H2O.AI AUTOML ИНИЦИАЛИЗИРОВАН:")
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
        labels = [item['sentiment'] for item in data]
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
            df['sentiment'] = labels

        # Конвертируем в H2O Frame
        h2o_frame = h2o.H2OFrame(df)

        if labels is not None:
            # Указываем что sentiment - категориальная переменная
            h2o_frame['sentiment'] = h2o_frame['sentiment'].asfactor()

        return h2o_frame

    def train(self, train_data, val_data=None):
        """
        Обучение H2O AutoML
        """
        print("🎯 АВТОМАТИЗИРОВАННЫЙ ПОИСК МОДЕЛЕЙ С H2O.AI...")

        # Подготовка данных
        X_train_texts = self.prepare_features(train_data)
        y_train = self.prepare_labels(train_data)

        # Проверяем классы
        unique_labels = set(y_train)
        print(f"📊 Обнаружены классы: {unique_labels}")
        print(f"📊 Размер данных: {len(X_train_texts)} примеров")

        # Создаем векторйзер и преобразуем данные
        print("📊 Векторизация текстов...")
        self.vectorizer.fit(X_train_texts)

        # Создаем H2O Frame для обучения
        print("📊 Создание H2O Frame...")
        train_frame = self._create_h2o_frame(X_train_texts, y_train)

        print(f"   Размерность: {train_frame.shape}")
        print(f"   Колонки: {train_frame.columns}")

        # Определяем features и target
        x = train_frame.columns[:-1]  # Все кроме последней колонки (target)
        y = 'sentiment'

        print(f"   Features: {len(x)} признаков")
        print(f"   Target: {y}")

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
        print(lb.head(rows=10))

        # Лучшая модель
        print(f"\n🎯 ЛУЧШАЯ МОДЕЛЬ:")
        print(f"   Алгоритм: {self.leader_model.algo}")
        print(f"   Model ID: {self.leader_model.model_id}")

        # Метрики лучшей модели
        try:
            performance = self.leader_model.model_performance()
            print(f"   AUC: {performance.auc():.4f}")
            print(f"   Logloss: {performance.logloss():.4f}")
            print(f"   Accuracy: {performance.accuracy():.4f}")
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
        prob_array = predictions[['p0', 'p1']].as_data_frame().values

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
        probabilities = prediction[['p0', 'p1']].as_data_frame().values[0]

        result = {
            'prediction': pred,
            'probabilities': {
                'class_0': f"{probabilities[0]:.3f}",
                'class_1': f"{probabilities[1]:.3f}"
            },
            'confidence': f"{max(probabilities):.3f}",
            'model_type': self.leader_model.algo
        }

        return result

    def evaluate(self, test_data):
        """
        Оценка на тестовых данных
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
        print(classification_report(y_test, predictions))

        return accuracy

    def get_model_info(self):
        """
        Возвращает информацию о моделях
        """
        if not self.is_trained:
            return {"error": "Модель не обучена"}

        lb = self.aml.leaderboard.as_data_frame()
        top_models = lb.head(5)[['model_id', 'auc', 'logloss']].to_dict('records')

        return {
            "leader_model": self.leader_model.model_id,
            "leader_algorithm": self.leader_model.algo,
            "top_models": top_models,
            "total_models_trained": len(lb),
            "feature_count": len(self.vectorizer.get_feature_names_out())
        }

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
            'model_path': model_path,
            'model_id': self.leader_model.model_id
        }, f"{filename}_meta.pkl")

        print(f"💾 Модель сохранена: {model_path}")

    def load_model(self, filename):
        """
        Загрузка модели
        """
        # Загружаем метаданные
        meta = joblib.load(f"{filename}_meta.pkl")
        self.vectorizer = meta['vectorizer']

        # Загружаем H2O модель
        self.leader_model = h2o.load_model(meta['model_path'])
        self.is_trained = True

        print(f"📥 Модель загружена: {meta['model_id']}")

    def __del__(self):
        """
        Закрытие H2O кластера при удалении объекта
        """
        try:
            h2o.cluster().shutdown()
        except:
            pass


# Упрощенная версия для быстрого старта
class SimpleH2OClassifier:
    """
    Упрощенная версия H2O классификатора
    """

    def __init__(self, time_seconds=120):
        h2o.init()

        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.aml = H2OAutoML(max_runtime_secs=time_seconds, seed=42)
        self.is_trained = False

        print(f"⚡ SimpleH2O: {time_seconds} сек")

    def train_and_test(self, train_data, test_data):
        """Обучает и тестирует за один шаг"""
        # Подготовка данных
        train_texts = [item['text'] for item in train_data]
        train_labels = [item['sentiment'] for item in train_data]
        test_texts = [item['text'] for item in test_data]
        test_labels = [item['sentiment'] for item in test_data]

        print(f"📚 Обучение на {len(train_texts)} примерах...")

        # Векторизация и создание H2O Frame
        self.vectorizer.fit(train_texts)

        train_features = self.vectorizer.transform(train_texts).toarray()
        test_features = self.vectorizer.transform(test_texts).toarray()

        # Создаем DataFrame
        feature_names = [f"f_{i}" for i in range(train_features.shape[1])]
        train_df = pd.DataFrame(train_features, columns=feature_names)
        train_df['target'] = train_labels

        test_df = pd.DataFrame(test_features, columns=feature_names)
        test_df['target'] = test_labels

        # H2O Frames
        train_frame = h2o.H2OFrame(train_df)
        test_frame = h2o.H2OFrame(test_df)
        train_frame['target'] = train_frame['target'].asfactor()
        test_frame['target'] = test_frame['target'].asfactor()

        # AutoML
        self.aml.train(x=feature_names, y='target', training_frame=train_frame)
        self.is_trained = True

        # Предсказания
        predictions = self.aml.leader.predict(test_frame)
        pred_array = predictions['predict'].as_data_frame().values.flatten()

        accuracy = accuracy_score(test_labels, pred_array)

        print(f"✅ Результаты:")
        print(f"   Лучшая модель: {self.aml.leader.algo}")
        print(f"   Точность: {accuracy:.4f}")

        return accuracy


# Основная функция
def main():
    """
    Основной пример использования H2O AutoML
    """
    # Загрузка данных
    train_data = read_jsonl_basic('../../util/news_sentiment_train.jsonl')
    test_data = read_jsonl_basic('../../util/news_sentiment_test.jsonl')

    print("=" * 50)
    print("🚀 H2O.AI AUTOML ДЛЯ КЛАССИФИКАЦИИ ТОНАЛЬНОСТИ")
    print("=" * 50)
    print(f"📁 Train: {len(train_data)} примеров")
    print(f"📁 Test: {len(test_data)} примеров")

    # Вариант 1: Полная версия H2O
    print("\n🎯 ВАРИАНТ 1: ПОЛНАЯ ВЕРСИЯ H2O AUTOML")
    h2o_model = H2OSentimentClassifier(max_runtime_secs=180)  # 3 минуты

    try:
        h2o_model.train(train_data)
        h2o_accuracy = h2o_model.evaluate(test_data)

        # Информация о моделях
        model_info = h2o_model.get_model_info()
        print(f"\n📋 ИНФОРМАЦИЯ О МОДЕЛЯХ:")
        print(f"   Лучшая модель: {model_info['leader_algorithm']}")
        print(f"   Всего моделей: {model_info['total_models_trained']}")
        print(f"   Топ-3 модели:")
        for i, model in enumerate(model_info['top_models'][:3]):
            print(f"      {i + 1}. {model['model_id']} (AUC: {model['auc']:.4f})")

    except Exception as e:
        print(f"❌ Ошибка H2O: {e}")
        h2o_accuracy = 0

    # Вариант 2: Упрощенная версия
    print("\n🎯 ВАРИАНТ 2: УПРОЩЕННАЯ ВЕРСИЯ")
    simple_h2o = SimpleH2OClassifier(time_seconds=120)  # 2 минуты

    try:
        simple_accuracy = simple_h2o.train_and_test(train_data, test_data)
    except Exception as e:
        print(f"❌ Ошибка упрощенной версии: {e}")
        simple_accuracy = 0

    # Демонстрация предсказаний
    if h2o_accuracy > 0:
        print("\n🧪 ДЕМОНСТРАЦИЯ ПРЕДСКАЗАНИЙ:")
        test_texts = [
            "Компания показала рекордные финансовые результаты",
            "Серьезные проблемы с поставками и качеством продукции",
            "Стабильный рост и положительные перспективы развития"
        ]

        for text in test_texts:
            result = h2o_model.predict_single(text)
            print(f"📝 '{text}'")
            print(f"   → Класс: {result['prediction']}")
            print(f"   → Уверенность: {result['confidence']}")
            print(f"   → Модель: {result['model_type']}")
            print()

        # Сохранение модели
        h2o_model.save_model("h2o_sentiment_model")

    # Завершение работы
    h2o.cluster().shutdown()
    print("✅ H2O кластер остановлен")


if __name__ == "__main__":
    main()