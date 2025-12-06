from catboost import CatBoostClassifier, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import json


class CatBoostMultiClassClassifier:
    """
    Многоклассовый классификатор на основе CatBoost
    """

    def __init__(self, iterations=1000, learning_rate=0.1, depth=6,
                 l2_leaf_reg=3, random_state=42, verbose=100):
        """
        Args:
            iterations: количество итераций
            learning_rate: скорость обучения
            depth: глубина деревьев
            l2_leaf_reg: L2 регуляризация
            random_state: для воспроизводимости
            verbose: вывод логов
        """
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            stop_words=None
        )

        # Для кодирования меток
        self.label_encoder = LabelEncoder()

        # CatBoost для многоклассовой классификации
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_state,
            verbose=verbose,
            loss_function='MultiClass',  # Многоклассовая классификация
            eval_metric='Accuracy',
            early_stopping_rounds=50,
            use_best_model=True
        )

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
        Обучение модели CatBoost для многоклассовой классификации
        """
        print("🎯 ОБУЧЕНИЕ CATBOOST (многоклассовый)...")

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

        # Преобразуем в плотный формат для CatBoost
        X_train_dense = X_train_vec.toarray()

        print(f"   Размерность признаков: {X_train_dense.shape}")
        print(f"   Количество классов: {self.n_classes}")
        print(f"   Количество примеров: {len(y_train)}")
        print(f"   Количество итераций: {self.model.get_param('iterations')}")
        print(f"   Глубина деревьев: {self.model.get_param('depth')}")
        print(f"   Learning rate: {self.model.get_param('learning_rate')}")

        # Подготовка данных для CatBoost
        if val_data:
            X_val, y_val = self.prepare_data(val_data)
            X_val_vec = self.vectorizer.transform(X_val)
            X_val_dense = X_val_vec.toarray()
            y_val_encoded = self.label_encoder.transform(y_val)

            train_pool = Pool(X_train_dense, label=y_train_encoded)
            val_pool = Pool(X_val_dense, label=y_val_encoded)

            print("🤖 Обучение CatBoost с валидацией...")
            self.model.fit(
                train_pool,
                eval_set=val_pool,
                plot=False,
                verbose=self.model.get_param('verbose')
            )
        else:
            train_pool = Pool(X_train_dense, label=y_train_encoded)
            print("🤖 Обучение CatBoost...")
            self.model.fit(train_pool)

        self.is_trained = True

        # Оценка на тренировочных данных
        train_pred = self.model.predict(X_train_dense)
        train_pred_decoded = self.label_encoder.inverse_transform(train_pred.flatten())
        train_accuracy = accuracy_score(y_train, train_pred_decoded)
        print(f"✅ Точность на train: {train_accuracy:.3f}")

        # Оценка на валидации, если есть
        if val_data:
            val_pred = self.model.predict(X_val_dense)
            val_pred_decoded = self.label_encoder.inverse_transform(val_pred.flatten())
            val_accuracy = accuracy_score(y_val, val_pred_decoded)
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

        predictions_encoded = self.model.predict(X_dense)
        predictions = self.label_encoder.inverse_transform(predictions_encoded.flatten())
        probabilities = self.model.predict_proba(X_dense)

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
            'model_type': 'CatBoost',
            'n_classes': self.n_classes
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
        Оценка модели на тестовых данных для многоклассовой классификации
        """
        X_test, y_test = self.prepare_data(test_data)
        X_test_vec = self.vectorizer.transform(X_test)
        X_test_dense = X_test_vec.toarray()

        # Кодируем тестовые метки
        y_test_encoded = self.label_encoder.transform(y_test)

        y_pred_encoded = self.model.predict(X_test_dense)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded.flatten())

        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)

        print("\n📊 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        print(classification_report(y_test, y_pred, digits=4))

        # Матрица ошибок
        print("\n📈 МАТРИЦА ОШИБОК:")
        cm = confusion_matrix(y_test_encoded, y_pred_encoded)
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

        # Точность по классам
        print(f"\n📊 Accuracy по классам:")
        for i, class_name in enumerate(class_names[:n_classes]):
            class_total = cm[i, :].sum()
            if class_total > 0:
                class_correct = cm[i, i]
                print(f"   {class_name}: {class_correct}/{class_total} ({class_correct / class_total:.1%})")

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
                min_len = min(len(feature_importances), len(feature_names))
                feature_importances = feature_importances[:min_len]
                feature_names = feature_names[:min_len]

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
            if total_importance > 0:
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
        print(f"   Количество классов: {self.n_classes}")

        # Получаем историю обучения
        if hasattr(self.model, 'get_evals_result'):
            try:
                evals_result = self.model.get_evals_result()
                if evals_result and 'learn' in evals_result:
                    learn_accuracy = evals_result['learn']['Accuracy'][-1] if 'Accuracy' in evals_result['learn'] else 0
                    print(f"   Final train accuracy: {learn_accuracy:.4f}")

                if evals_result and 'validation' in evals_result:
                    val_accuracy = evals_result['validation']['Accuracy'][-1] if 'Accuracy' in evals_result[
                        'validation'] else 0
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
            'class_names': self.class_names,
            'n_classes': self.n_classes
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
        self.label_encoder = loaded['label_encoder']
        self.class_names = loaded.get('class_names', [])
        self.n_classes = loaded.get('n_classes', 0)
        self.is_trained = True
        print(f"📥 Модель CatBoost загружена: {filename}")


# СПЕЦИАЛИЗИРОВАННЫЙ КЛАССИФИКАТОР ДЛЯ ТОНАЛЬНОСТИ (3 КЛАССА)
class SentimentCatBoostClassifier(CatBoostMultiClassClassifier):
    """
    Специализированный CatBoost классификатор для тональности (негатив, нейтрал, позитив)
    """

    def __init__(self, iterations=1000, learning_rate=0.1, depth=6,
                 l2_leaf_reg=3, random_state=42, verbose=100):
        super().__init__(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_state=random_state,
            verbose=verbose
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


# СРАВНЕНИЕ РАЗНЫХ ПАРАМЕТРОВ CATBOOST
def compare_catboost_parameters_multiclass(train_data, val_data):
    """
    Сравнение разных параметров CatBoost для многоклассовой классификации
    """
    print("🔬 СРАВНЕНИЕ ПАРАМЕТРОВ CATBOOST (многоклассовый)")
    print("=" * 50)

    models = {}

    # 1. Разная глубина деревьев
    for depth in [4, 6, 8]:
        print(f"\n1. CatBoost с depth={depth}:")
        model = CatBoostMultiClassClassifier(depth=depth, iterations=300, verbose=0)
        # Используем try-except для обработки ошибок
        try:
            model.train(train_data, val_data)
            models[f'CB_depth_{depth}'] = model
        except Exception as e:
            print(f"   ❌ Ошибка при обучении: {e}")
            continue

    # 2. Разный learning rate
    for lr in [0.05, 0.1, 0.2]:
        print(f"\n2. CatBoost с learning_rate={lr}:")
        model = CatBoostMultiClassClassifier(learning_rate=lr, iterations=300, verbose=0)
        try:
            model.train(train_data, val_data)
            models[f'CB_lr_{lr}'] = model
        except Exception as e:
            print(f"   ❌ Ошибка при обучении: {e}")
            continue

    return models


# АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ С ГРУППИРОВКОЙ
def analyze_catboost_features_multiclass(model, top_n=30):
    """
    Детальный анализ важности признаков для CatBoost (многоклассовый)
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

        # Ключевые слова для разных тональностей (адаптируйте под ваши классы)
        if hasattr(model, 'class_names'):
            if 'positive' in model.class_names or 'pos' in model.class_names:
                positive_keywords = ['хорош', 'отлич', 'прекрас', 'довол', 'рекоменд', 'великол', 'замечат']
            else:
                positive_keywords = []

            if 'negative' in model.class_names or 'neg' in model.class_names:
                negative_keywords = ['плох', 'ужас', 'разочар', 'недовол', 'проблем', 'кошмар', 'некачеств']
            else:
                negative_keywords = []

        for item in importance_data:
            feature = item['feature']
            if any(keyword in feature for keyword in positive_keywords):
                positive_words.append(item)
            elif any(keyword in feature for keyword in negative_keywords):
                negative_words.append(item)
            else:
                neutral_words.append(item)

        if positive_words:
            print(f"\n🎯 ПОЛОЖИТЕЛЬНЫЕ ПРИЗНАКИ:")
            for item in positive_words[:10]:
                print(f"   {item['feature']}: {item['importance']:.4f}")

        if negative_words:
            print(f"\n🎯 ОТРИЦАТЕЛЬНЫЕ ПРИЗНАКИ:")
            for item in negative_words[:10]:
                print(f"   {item['feature']}: {item['importance']:.4f}")

        if neutral_words:
            print(f"\n🎯 НЕЙТРАЛЬНЫЕ/СМЕШАННЫЕ ПРИЗНАКИ:")
            for item in neutral_words[:10]:
                print(f"   {item['feature']}: {item['importance']:.4f}")


# ПРИМЕР ИСПОЛЬЗОВАНИЯ
def main():
    """
    Пример использования CatBoost для многоклассовой классификации
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

    # Создаем и обучаем CatBoost
    print("\n" + "=" * 50)
    print("🎯 ОБУЧЕНИЕ CATBOOST (многоклассовый)")
    print("=" * 50)

    cb_classifier = CatBoostMultiClassClassifier(
        iterations=500,  # Уменьшено для скорости
        learning_rate=0.1,
        depth=6,
        verbose=100
    )

    # Показываем распределение классов
    print("\n📊 РАСПРЕДЕЛЕНИЕ КЛАССОВ В ОБУЧАЮЩИХ ДАННЫХ:")
    class_dist = cb_classifier.get_class_distribution(train_subset)
    for class_name, count in class_dist.items():
        print(f"   {class_name}: {count} примеров ({count / len(train_subset):.1%})")

    cb_classifier.train(train_subset, val_subset)

    # Детальный анализ признаков
    analyze_catboost_features_multiclass(cb_classifier, top_n=25)

    # Оценка на тестовых данных
    print("\n🧪 ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ...")
    test_accuracy = cb_classifier.evaluate(test_data)

    # Сохранение модели
    cb_classifier.save_model("catboost_multiclass_model.pkl")

    # Сравнение параметров (если есть достаточно данных)
    print("\n" + "=" * 50)
    print("🔬 СРАВНЕНИЕ ПАРАМЕТРОВ")
    print("=" * 50)

    # Используем меньше данных для быстрого сравнения
    small_train = train_subset[:60]
    small_val = val_subset[:15]

    # Проверяем, есть ли достаточно данных
    if len(small_train) >= 10 and len(small_val) >= 5:
        models = compare_catboost_parameters_multiclass(small_train, small_val)
    else:
        print("⚠️  Недостаточно данных для сравнения параметров")

    return cb_classifier


if __name__ == "__main__":
    print("🚀 ЗАПУСК МНОГОКЛАССОВОГО CATBOOST КЛАССИФИКАТОРА")
    print("=" * 80)

    # Запускаем основной пример
    classifier = main()