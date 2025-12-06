from catboost import CatBoostClassifier, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import joblib
import pandas as pd

from util.jsonl_process import read_jsonl_basic


class CatBoostSentimentClassifier:
    """
    Бинарный классификатор тональности на основе CatBoost
    """

    def __init__(self, iterations=1000, learning_rate=0.1, depth=6,
                 l2_leaf_reg=3, random_state=42, verbose=100,
                 positive_label=1, negative_label=0,
                 text_processing='tfidf'):
        """
        Args:
            iterations: количество итераций
            learning_rate: скорость обучения
            depth: глубина деревьев
            l2_leaf_reg: L2 регуляризация
            random_state: для воспроизводимости
            verbose: вывод логов
            positive_label: метка положительного класса
            negative_label: метка отрицательного класса
            text_processing: 'tfidf' или 'bow' для обработки текста
        """
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            stop_words=None
        )

        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_state,
            verbose=verbose,
            loss_function='Logloss',
            eval_metric='Accuracy',
            early_stopping_rounds=50,
            use_best_model=True
        )

        self.positive_label = positive_label
        self.negative_label = negative_label
        self.text_processing = text_processing
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
        Обучение модели CatBoost
        """
        print("🎯 ОБУЧЕНИЕ CATBOOST...")

        # Подготовка данных
        X_train, y_train = self.prepare_data(train_data)

        # Проверяем, что у нас только 2 класса
        unique_labels = set(y_train)
        if len(unique_labels) != 2:
            print(f"⚠️  Предупреждение: обнаружено {len(unique_labels)} классов: {unique_labels}")

        # Векторизация текстов
        print("📊 Векторизация текстов...")
        X_train_vec = self.vectorizer.fit_transform(X_train)

        # Преобразуем в плотный формат для CatBoost
        X_train_dense = X_train_vec.toarray()

        print(f"   Размерность признаков: {X_train_dense.shape}")
        print(f"   Классы: {unique_labels}")
        print(f"   Количество итераций: {self.model.get_param('iterations')}")
        print(f"   Глубина деревьев: {self.model.get_param('depth')}")
        print(f"   Learning rate: {self.model.get_param('learning_rate')}")

        # Подготовка данных для CatBoost
        if val_data:
            X_val, y_val = self.prepare_data(val_data)
            X_val_vec = self.vectorizer.transform(X_val)
            X_val_dense = X_val_vec.toarray()

            train_pool = Pool(X_train_dense, label=y_train)
            val_pool = Pool(X_val_dense, label=y_val)

            print("🤖 Обучение CatBoost с валидацией...")
            self.model.fit(
                train_pool,
                eval_set=val_pool,
                plot=False,
                verbose=self.model.get_param('verbose')
            )
        else:
            train_pool = Pool(X_train_dense, label=y_train)
            print("🤖 Обучение CatBoost...")
            self.model.fit(train_pool)

        self.is_trained = True

        # Оценка на тренировочных данных
        train_pred = self.model.predict(X_train_dense)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"✅ Точность на train: {train_accuracy:.3f}")

        # Оценка на валидации, если есть
        if val_data:
            val_pred = self.model.predict(X_val_dense)
            val_accuracy = accuracy_score(y_val, val_pred)
            print(f"✅ Точность на val: {val_accuracy:.3f}")

        # Покажем важные признаки
        self._show_important_features(top_n=20)

        # Информация о модели
        self._show_model_info()

    def predict(self, texts):
        """
        Предсказание для списка текстов
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)
        X_dense = X_vec.toarray()

        predictions = self.model.predict(X_dense)
        probabilities = self.model.predict_proba(X_dense)

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

    def predict_proba(self, texts):
        """
        Только вероятности для текстов
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)
        X_dense = X_vec.toarray()

        return self.model.predict_proba(X_dense)

    def evaluate(self, test_data):
        """
        Оценка модели на тестовых данных
        """
        X_test, y_test = self.prepare_data(test_data)
        X_test_vec = self.vectorizer.transform(X_test)
        X_test_dense = X_test_vec.toarray()

        y_pred = self.model.predict(X_test_dense)
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

    def _show_important_features(self, top_n=20):
        """
        Показывает самые важные признаки
        """
        try:
            feature_importances = self.model.get_feature_importance()
            feature_names = self.vectorizer.get_feature_names_out()

            # Убедимся, что размерности совпадают
            if len(feature_importances) != len(feature_names):
                print(
                    f"⚠️  Размерность важностей ({len(feature_importances)}) не совпадает с количеством признаков ({len(feature_names)})")
                # Берем только первые n важностей
                feature_importances = feature_importances[:len(feature_names)]

            print(f"\n🔍 ТОП-{top_n} ВАЖНЫХ ПРИЗНАКОВ (CatBoost):")

            # Сортируем признаки по важности
            indices = np.argsort(feature_importances)[::-1]

            print(f"\n   САМЫЕ ВАЖНЫЕ ПРИЗНАКИ:")
            for i in range(min(top_n, len(indices))):
                idx = indices[i]
                print(f"      {feature_names[idx]}: {feature_importances[idx]:.4f}")

            # Статистика важности
            total_importance = np.sum(feature_importances)
            top_n_importance = np.sum(feature_importances[indices[:top_n]])
            print(
                f"\n   📊 Топ-{top_n} признаков объясняют {top_n_importance / total_importance * 100:.1f}% общей важности")

        except Exception as e:
            print(f"❌ Ошибка при получении важности признаков: {e}")

    def _show_model_info(self):
        """
        Показывает информацию о обученной модели
        """
        print(f"\n📊 ИНФОРМАЦИЯ О CATBOOST МОДЕЛИ:")
        print(f"   Количество деревьев: {self.model.tree_count_}")

        # Получаем историю обучения
        if hasattr(self.model, 'get_evals_result'):
            try:
                evals_result = self.model.get_evals_result()
                if evals_result and 'learn' in evals_result:
                    learn_accuracy = evals_result['learn']['Accuracy'][-1]
                    print(f"   Final train accuracy: {learn_accuracy:.4f}")

                if evals_result and 'validation' in evals_result:
                    val_accuracy = evals_result['validation']['Accuracy'][-1]
                    print(f"   Final validation accuracy: {val_accuracy:.4f}")
            except:
                pass

        # Best iteration
        try:
            best_iteration = self.model.get_best_iteration()
            print(f"   Best iteration: {best_iteration}")
        except:
            pass

    def get_feature_importance_df(self, top_n=50):
        """
        Возвращает DataFrame с важностью признаков
        """
        try:
            feature_importances = self.model.get_feature_importance()
            feature_names = self.vectorizer.get_feature_names_out()

            # Обрезаем до минимальной длины
            min_len = min(len(feature_importances), len(feature_names))
            feature_importances = feature_importances[:min_len]
            feature_names = feature_names[:min_len]

            # Сортируем по важности
            indices = np.argsort(feature_importances)[::-1]

            importance_data = []
            for i in range(min(top_n, len(indices))):
                idx = indices[i]
                importance_data.append({
                    'feature': feature_names[idx],
                    'importance': feature_importances[idx],
                    'rank': i + 1
                })

            return importance_data
        except Exception as e:
            print(f"❌ Ошибка при создании DataFrame важности: {e}")
            return None

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
        print(f"💾 Модель CatBoost сохранена: {filename}")

        # Также сохраняем в native CatBoost format
        cb_filename = filename.replace('.pkl', '.cbm')
        self.model.save_model(cb_filename)
        print(f"💾 Модель CatBoost сохранена (native): {cb_filename}")

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
        print(f"📥 Модель CatBoost загружена: {filename}")


# Сравнение разных параметров CatBoost
def compare_catboost_parameters(train_data, val_data):
    """
    Сравнение разных параметров CatBoost
    """
    print("🔬 СРАВНЕНИЕ ПАРАМЕТРОВ CATBOOST")
    print("=" * 50)

    models = {}

    # 1. Разная глубина деревьев
    for depth in [4, 6, 8]:
        print(f"\n1. CatBoost с depth={depth}:")
        model = CatBoostSentimentClassifier(depth=depth, iterations=500, verbose=0)
        model.train(train_data, val_data)
        models[f'CB_depth_{depth}'] = model

    # 2. Разный learning rate
    for lr in [0.05, 0.1, 0.2]:
        print(f"\n2. CatBoost с learning_rate={lr}:")
        model = CatBoostSentimentClassifier(learning_rate=lr, iterations=500, verbose=0)
        model.train(train_data, val_data)
        models[f'CB_lr_{lr}'] = model

    return models


# Анализ важности признаков с группировкой
def analyze_catboost_features(model, top_n=30):
    """
    Детальный анализ важности признаков для CatBoost
    """
    importance_data = model.get_feature_importance_df(top_n=top_n)

    if importance_data:
        print(f"\n📈 ДЕТАЛЬНЫЙ АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ CATBOOST (Топ-{top_n}):")
        print("=" * 60)

        for i, item in enumerate(importance_data[:top_n]):
            print(f"{i + 1:2d}. {item['feature']:20s} : {item['importance']:.4f}")

        # Группируем по типам признаков
        positive_words = []
        negative_words = []
        neutral_words = []

        positive_keywords = ['хорош', 'отлич', 'прекрас', 'довол', 'рекоменд', 'великол', 'замечат']
        negative_keywords = ['плох', 'ужас', 'разочар', 'недовол', 'проблем', 'кошмар', 'некачеств']

        for item in importance_data:
            feature = item['feature']
            if any(keyword in feature for keyword in positive_keywords):
                positive_words.append(item)
            elif any(keyword in feature for keyword in negative_keywords):
                negative_words.append(item)
            else:
                neutral_words.append(item)

        print(f"\n🎯 ПОЛОЖИТЕЛЬНЫЕ ПРИЗНАКИ:")
        for item in positive_words[:10]:
            print(f"   {item['feature']}: {item['importance']:.4f}")

        print(f"\n🎯 ОТРИЦАТЕЛЬНЫЕ ПРИЗНАКИ:")
        for item in negative_words[:10]:
            print(f"   {item['feature']}: {item['importance']:.4f}")

        print(f"\n🎯 НЕЙТРАЛЬНЫЕ/СМЕШАННЫЕ ПРИЗНАКИ:")
        for item in neutral_words[:10]:
            print(f"   {item['feature']}: {item['importance']:.4f}")


# Пример использования
def main():
    """
    Пример использования CatBoost для классификации тональности
    """
    train_data = read_jsonl_basic('../../util/news_sentiment_train.jsonl')
    val_data = read_jsonl_basic('../../util/news_sentiment_val.jsonl')
    test_data = read_jsonl_basic('../../util/news_sentiment_test.jsonl')

    print(f"📊 Данные: {len(train_data)} train, {len(val_data)} val")

    # Обучаем CatBoost
    print("\n" + "=" * 50)
    cb_classifier = CatBoostSentimentClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        verbose=100
    )
    cb_classifier.train(train_data, val_data)

    # Детальный анализ признаков
    analyze_catboost_features(cb_classifier, top_n=25)

    # Сохранение модели
    cb_classifier.save_model("catboost_sentiment_classifier.pkl")


if __name__ == "__main__":
    main()