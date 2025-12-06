from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from util.jsonl_process import read_jsonl_basic


class MultiCategoryClassifier:
    """
    Многоклассовый классификатор категорий на основе логистической регрессии
    """

    def __init__(self,
                 regularization: str = 'l2',
                 C: float = 1.0,
                 class_names: Optional[List[str]] = None,
                 solver: str = 'lbfgs',
                 max_iter: int = 1000,
                 text_field: str = 'text',
                 label_field: str = 'category'):
        """
        Args:
            regularization: 'l1' или 'l2' регуляризация
            C: параметр регуляризации (меньше = сильнее регуляризация)
            class_names: список названий классов (опционально)
            solver: алгоритм оптимизации ('lbfgs', 'newton-cg', 'saga', 'sag')
            max_iter: максимальное количество итераций
            text_field: название поля с текстом
            label_field: название поля с меткой категории
        """
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            stop_words=None
        )

        # Для многоклассовой классификации используем multinomial логистическую регрессию
        self.model = LogisticRegression(
            penalty=regularization,
            C=C,
            random_state=42,
            solver=solver,
            max_iter=max_iter,
            multi_class='multinomial'
        )

        self.label_encoder = LabelEncoder()
        self.class_names = class_names
        self.is_trained = False
        self.num_classes = 0
        self.class_mapping = {}
        self.text_field = text_field
        self.label_field = label_field

    def prepare_data(self, data: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        Подготовка данных: извлекаем тексты и метки категорий
        """
        texts = [item[self.text_field] for item in data]
        labels = [item[self.label_field] for item in data]
        return texts, labels

    def analyze_class_distribution(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Анализ распределения категорий в данных
        """
        _, labels = self.prepare_data(data)
        label_counts = Counter(labels)

        result = {
            'total_samples': len(data),
            'num_classes': len(label_counts),
            'classes': dict(label_counts),
            'class_percentages': {},
            'imbalance_ratio': None
        }

        if label_counts:
            max_count = max(label_counts.values())
            min_count = min(label_counts.values())
            if min_count > 0:
                result['imbalance_ratio'] = max_count / min_count

            for label, count in label_counts.items():
                result['class_percentages'][label] = count / len(data) * 100

        return result

    def train(self, train_data: List[Dict[str, Any]],
              val_data: Optional[List[Dict[str, Any]]] = None,
              auto_detect_classes: bool = True,
              handle_imbalance: bool = True) -> None:
        """
        Обучение модели
        """
        print("🎯 ОБУЧЕНИЕ МНОГОКЛАССОВОГО КЛАССИФИКАТОРА КАТЕГОРИЙ...")
        print(f"   Поле с текстом: '{self.text_field}'")
        print(f"   Поле с категорией: '{self.label_field}'")

        # Анализ распределения категорий
        train_dist = self.analyze_class_distribution(train_data)
        print(f"\n📊 РАСПРЕДЕЛЕНИЕ КАТЕГОРИЙ В TRAIN:")
        print(f"   Всего примеров: {train_dist['total_samples']}")
        print(f"   Количество категорий: {train_dist['num_classes']}")

        if train_dist['imbalance_ratio']:
            print(f"   Коэффициент дисбаланса: {train_dist['imbalance_ratio']:.2f}")
            if train_dist['imbalance_ratio'] > 3:
                print("   ⚠️  Обнаружен сильный дисбаланс категорий")
                if handle_imbalance:
                    print("   ✅ Будет применено взвешивание классов")
                    self.model.set_params(class_weight='balanced')
            else:
                print("   ✅ Дисбаланс в пределах нормы")

        # Подготовка данных
        X_train, y_train_raw = self.prepare_data(train_data)

        # Кодируем метки
        if auto_detect_classes:
            self.label_encoder.fit(y_train_raw)
            y_train = self.label_encoder.transform(y_train_raw)
            self.class_names = list(self.label_encoder.classes_)
        else:
            if self.class_names is None:
                raise ValueError("class_names должен быть задан, если auto_detect_classes=False")
            self.label_encoder.fit(self.class_names)
            y_train = self.label_encoder.transform(y_train_raw)

        self.num_classes = len(self.class_names)
        self.class_mapping = {i: cls for i, cls in enumerate(self.class_names)}

        # Выводим информацию о категориях
        print(f"\n📋 СПИСОК КАТЕГОРИЙ ({self.num_classes}):")
        for i, (class_name, count) in enumerate(train_dist['classes'].items()):
            percentage = train_dist['class_percentages'].get(class_name, 0)
            print(f"   {i + 1:2d}. {class_name}: {count} примеров ({percentage:.1f}%)")

        # Векторизация текстов
        print("\n📊 Векторизация текстов...")
        X_train_vec = self.vectorizer.fit_transform(X_train)

        print(f"   Размерность признаков: {X_train_vec.shape}")
        print(f"   Уникальных слов/фраз: {len(self.vectorizer.get_feature_names_out())}")
        print(f"   Плотность матрицы: {X_train_vec.nnz / (X_train_vec.shape[0] * X_train_vec.shape[1]):.4f}")

        # Обучение модели
        print("\n🤖 Обучение модели...")
        self.model.fit(X_train_vec, y_train)
        self.is_trained = True

        # Оценка на тренировочных данных
        train_pred = self.model.predict(X_train_vec)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"\n✅ Точность на train: {train_accuracy:.3f}")

        # Отчет по классам на train
        print("\n📊 ОТЧЕТ ПО КАТЕГОРИЯМ (train):")
        print(classification_report(y_train, train_pred, target_names=self.class_names))

        # Оценка на валидации, если есть
        if val_data:
            val_accuracy, _ = self.evaluate(val_data, detailed=False)
            print(f"✅ Точность на val: {val_accuracy:.3f}")

        # Покажем важные признаки для каждой категории
        self._show_important_features(top_n=12)

        # Матрица ошибок на train
        self._plot_confusion_matrix(y_train, train_pred, "Train Confusion Matrix")

    def predict(self, texts: List[str]) -> Tuple[List[str], np.ndarray]:
        """
        Предсказание для списка текстов
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)
        predictions_encoded = self.model.predict(X_vec)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        probabilities = self.model.predict_proba(X_vec)

        return predictions, probabilities

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Предсказание для одного текста с детальной информацией
        """
        predictions, probabilities = self.predict([text])
        pred = predictions[0]
        pred_encoded = self.label_encoder.transform([pred])[0]
        prob = probabilities[0]

        # Получаем вероятности для всех категорий
        class_probs = {}
        for i, cls in enumerate(self.class_names):
            class_probs[cls] = prob[i]

        # Находим топ-3 наиболее вероятных категории
        top_n = min(3, self.num_classes)
        top_indices = np.argsort(prob)[-top_n:][::-1]
        top_classes = []
        for idx in top_indices:
            top_classes.append({
                'category': self.class_names[idx],
                'probability': prob[idx],
                'probability_percent': prob[idx] * 100
            })

        return {
            'prediction': pred,
            'category': pred,  # для совместимости
            'prediction_encoded': pred_encoded,
            'category_probabilities': class_probs,
            'top_categories': top_classes,
            'confidence': prob[pred_encoded],
            'confidence_percent': prob[pred_encoded] * 100
        }

    def evaluate(self, test_data: List[Dict[str, Any]],
                 detailed: bool = True,
                 plot_confusion_matrix: bool = True) -> Tuple[float, Dict]:
        """
        Оценка модели на тестовых данных
        """
        X_test, y_test_raw = self.prepare_data(test_data)
        y_test = self.label_encoder.transform(y_test_raw)
        X_test_vec = self.vectorizer.transform(X_test)

        y_pred_encoded = self.model.predict(X_test_vec)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        accuracy = accuracy_score(y_test, y_pred_encoded)

        # Анализ распределения в тестовых данных
        test_dist = self.analyze_class_distribution(test_data)

        if detailed:
            print(f"\n📊 ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ:")
            print(f"   Примеров: {len(test_data)}")
            print(f"   Категорий: {test_dist['num_classes']}")
            print(f"   Точность: {accuracy:.3f}")

            print(f"\n📈 ДЕТАЛЬНЫЙ ОТЧЕТ ПО КАТЕГОРИЯМ:")
            print(classification_report(y_test, y_pred_encoded, target_names=self.class_names, digits=3))

            # Матрица ошибок
            print(f"\n📊 МАТРИЦА ОШИБОК:")
            cm = confusion_matrix(y_test, y_pred_encoded)
            self._print_confusion_matrix(cm)

            if plot_confusion_matrix:
                self._plot_confusion_matrix(y_test, y_pred_encoded, "Test Confusion Matrix")

        # Дополнительные метрики
        report_dict = classification_report(y_test, y_pred_encoded,
                                            target_names=self.class_names,
                                            output_dict=True)

        return accuracy, report_dict

    def _print_confusion_matrix(self, cm: np.ndarray) -> None:
        """
        Красиво печатает матрицу ошибок
        """
        n_classes = len(self.class_names)

        # Заголовок
        max_class_len = max(len(cls) for cls in self.class_names)
        header_padding = max(12, max_class_len + 2)

        header = " " * header_padding + " | "
        header += " ".join([f"{cls[:10]:>10}" for cls in self.class_names])
        print(header)
        print("-" * (header_padding + 3 + n_classes * 11))

        # Строки
        for i, cls in enumerate(self.class_names):
            row = f"{cls[:header_padding - 2]:>{header_padding - 2}} | "
            row += " ".join([f"{cm[i][j]:>10}" for j in range(n_classes)])
            print(row)

    def _plot_confusion_matrix(self, y_true: np.ndarray,
                               y_pred: np.ndarray,
                               title: str = "Confusion Matrix") -> None:
        """
        Визуализация матрицы ошибок
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(max(10, self.num_classes), max(8, self.num_classes * 0.8)))

            # Нормализуем по строкам (по истинным меткам)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=self.class_names,
                        yticklabels=self.class_names,
                        vmin=0, vmax=1)
            plt.title(f"{title} (нормализована)")
            plt.ylabel('Истинные категории')
            plt.xlabel('Предсказанные категории')
            plt.tight_layout()

            # Сохраняем с разными разрешениями
            filename = title.lower().replace(' ', '_').replace('-', '_')
            plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{filename}_highres.png", dpi=600, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"⚠️  Не удалось построить матрицу ошибок: {e}")

    def _show_important_features(self, top_n: int = 12) -> None:
        """
        Показывает самые важные признаки для каждой категории
        """
        if not hasattr(self.model, 'coef_'):
            return

        feature_names = self.vectorizer.get_feature_names_out()

        print(f"\n🔍 ТОП-{top_n} ВАЖНЫХ ПРИЗНАКОВ ДЛЯ КАЖДОЙ КАТЕГОРИИ:")

        # Для многоклассовой классификации у нас отдельные коэффициенты для каждого класса
        for class_idx, class_name in enumerate(self.class_names):
            print(f"\n   КАТЕГОРИЯ '{class_name}':")
            coef = self.model.coef_[class_idx]

            # Положительные признаки (указывают на данную категорию)
            pos_indices = np.argsort(coef)[-top_n:][::-1]
            print(f"      Показатели ДЛЯ категории:")
            for idx in pos_indices:
                print(f"        + {feature_names[idx]}: {coef[idx]:.3f}")

            # Отрицательные признаки (указывают ПРОТИВ данной категории)
            neg_indices = np.argsort(coef)[:top_n]
            print(f"\n      Показатели ПРОТИВ категории:")
            for idx in neg_indices:
                print(f"        - {feature_names[idx]}: {coef[idx]:.3f}")

    def predict_proba_for_category(self, texts: List[str], target_category: str) -> np.ndarray:
        """
        Возвращает вероятности для конкретной категории
        """
        if target_category not in self.class_names:
            raise ValueError(f"Категория {target_category} не найдена. Доступные категории: {self.class_names}")

        _, probabilities = self.predict(texts)
        category_idx = list(self.class_names).index(target_category)

        return probabilities[:, category_idx]

    def get_class_distribution(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Получить распределение категорий в данных
        """
        return self.analyze_class_distribution(data)

    def save_model(self, filename: str) -> None:
        """
        Сохранение модели
        """
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'class_names': self.class_names,
            'class_mapping': self.class_mapping,
            'num_classes': self.num_classes,
            'text_field': self.text_field,
            'label_field': self.label_field
        }, filename)
        print(f"💾 Модель сохранена: {filename}")

    def load_model(self, filename: str) -> None:
        """
        Загрузка модели
        """
        loaded = joblib.load(filename)
        self.model = loaded['model']
        self.vectorizer = loaded['vectorizer']
        self.label_encoder = loaded['label_encoder']
        self.class_names = loaded['class_names']
        self.class_mapping = loaded['class_mapping']
        self.num_classes = loaded['num_classes']
        self.text_field = loaded.get('text_field', 'text')
        self.label_field = loaded.get('label_field', 'category')
        self.is_trained = True

        print(f"📥 Модель загружена: {filename}")
        print(f"   Категории: {self.class_names}")
        print(f"   Количество категорий: {self.num_classes}")
        print(f"   Поле с текстом: '{self.text_field}'")
        print(f"   Поле с категорией: '{self.label_field}'")

    def predict_batch_with_details(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Предсказание для батча текстов с детальной информацией
        """
        predictions, probabilities = self.predict(texts)

        results = []
        for i, (text, pred, probs) in enumerate(zip(texts, predictions, probabilities)):
            pred_encoded = self.label_encoder.transform([pred])[0]

            # Находим топ-3 категории
            top_n = min(3, self.num_classes)
            top_indices = np.argsort(probs)[-top_n:][::-1]
            top_categories = []
            for idx in top_indices:
                top_categories.append({
                    'category': self.class_names[idx],
                    'probability': probs[idx],
                    'probability_percent': probs[idx] * 100
                })

            results.append({
                'text': text,
                'prediction': pred,
                'predicted_category': pred,
                'confidence': probs[pred_encoded],
                'confidence_percent': probs[pred_encoded] * 100,
                'top_categories': top_categories,
                'all_probabilities': {cls: probs[i] for i, cls in enumerate(self.class_names)}
            })

        return results


# Функция для быстрого обучения
def quick_train(train_file: str,
                val_file: Optional[str] = None,
                test_file: Optional[str] = None,
                text_field: str = 'text',
                label_field: str = 'category',
                output_model: str = 'category_classifier.pkl') -> MultiCategoryClassifier:
    """
    Быстрое обучение многоклассового классификатора из файлов
    """
    import json
    import os

    def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
        if not os.path.exists(filepath):
            print(f"⚠️  Файл не найден: {filepath}")
            return []
        with open(filepath, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    print("🚀 ЗАПУСК БЫСТРОГО ОБУЧЕНИЯ КЛАССИФИКАТОРА КАТЕГОРИЙ")
    print("=" * 60)

    # Загрузка данных
    print(f"\n📥 Загрузка данных...")
    train_data = load_jsonl(train_file)
    if not train_data:
        print(f"❌ Ошибка: не удалось загрузить тренировочные данные из {train_file}")
        return None

    print(f"   Train: {len(train_data)} примеров")

    if val_file:
        val_data = load_jsonl(val_file)
        print(f"   Val: {len(val_data)} примеров")
    else:
        val_data = None
        print(f"   Val: не указан")

    if test_file:
        test_data = load_jsonl(test_file)
        print(f"   Test: {len(test_data)} примеров")
    else:
        test_data = None

    # Проверяем структуру данных
    if train_data:
        sample_item = train_data[0]
        print(f"\n📋 СТРУКТУРА ДАННЫХ:")
        print(f"   Поля в данных: {list(sample_item.keys())}")
        print(f"   Используем поле текста: '{text_field}'")
        print(f"   Используем поле категории: '{label_field}'")

        if text_field not in sample_item:
            print(f"❌ Ошибка: поле '{text_field}' не найдено в данных")
            return None
        if label_field not in sample_item:
            print(f"❌ Ошибка: поле '{label_field}' не найдено в данных")
            return None

    # Обучаем модель
    print(f"\n🎯 Начало обучения...")
    classifier = MultiCategoryClassifier(
        regularization='l2',
        C=1.0,
        class_names=None,  # определим автоматически
        solver='lbfgs',
        max_iter=1000,
        text_field=text_field,
        label_field=label_field
    )

    classifier.train(train_data, val_data, auto_detect_classes=True, handle_imbalance=True)

    # Тестируем, если есть тестовые данные
    if test_data:
        print(f"\n🧪 Тестирование на тестовых данных...")
        accuracy, report = classifier.evaluate(test_data, detailed=True)
        print(f"\n🎯 Итоговая точность на тесте: {accuracy:.3f}")

        # Сохраняем отчет
        import pandas as pd
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv('classification_report.csv', index=True)
        print(f"📄 Детальный отчет сохранен в 'classification_report.csv'")

    # Сохраняем модель
    classifier.save_model(output_model)

    # Тестовый пример
    print(f"\n🧪 ТЕСТОВЫЙ ПРИМЕР РАБОТЫ МОДЕЛИ:")
    if train_data:
        sample_text = train_data[0][text_field][:100] + "..." if len(train_data[0][text_field]) > 100 else \
        train_data[0][text_field]
        result = classifier.predict_single(sample_text)
        print(f"   Текст: '{sample_text}'")
        print(f"   Предсказанная категория: {result['prediction']}")
        print(f"   Уверенность: {result['confidence_percent']:.1f}%")

        if result['top_categories']:
            print(f"   Топ-3 категории:")
            for i, cat in enumerate(result['top_categories'], 1):
                print(f"     {i}. {cat['category']}: {cat['probability_percent']:.1f}%")

    return classifier


def main():
    """
    Пример использования классификатора категорий
    """
    # Загрузка данных (предполагается формат с полями 'text' и 'category')
    train_data = read_jsonl_basic('../util/news_category_train.jsonl')
    val_data = read_jsonl_basic('../util/news_category_val.jsonl')
    test_data = read_jsonl_basic('../util/news_category_test.jsonl')

    print(f"📊 Данные: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Проверяем структуру данных
    if train_data:
        print(f"\n📋 ПРИМЕР ДАННЫХ:")
        print(f"   Поля: {list(train_data[0].keys())}")
        print(f"   Пример текста: {train_data[0].get('text', 'N/A')[:100]}...")
        print(f"   Категория: {train_data[0].get('category', 'N/A')}")

    # Определяем названия полей
    text_field = 'text'
    label_field = 'category'

    # 1. Обучаем модель
    print("\n" + "=" * 60)

    classifier = MultiCategoryClassifier(
        regularization='l2',
        C=1.0,
        class_names=None,  # определим автоматически
        solver='lbfgs',
        max_iter=1000,
        text_field=text_field,
        label_field=label_field
    )

    classifier.train(train_data, val_data, auto_detect_classes=True, handle_imbalance=True)

    # 2. Оценка на тестовых данных
    if test_data:
        print("\n" + "=" * 60)
        test_accuracy, test_report = classifier.evaluate(test_data, detailed=True)
        print(f"\n📊 Итоговая точность на тесте: {test_accuracy:.3f}")

    # 4. Сохраняем модель
    classifier.save_model("multiclass_category_classifier.pkl")

    # 5. Демонстрация загрузки модели
    print("\n" + "=" * 60)
    print("🔄 ДЕМОНСТРАЦИЯ ЗАГРУЗКИ МОДЕЛИ:")
    loaded_classifier = MultiCategoryClassifier()
    loaded_classifier.load_model("multiclass_category_classifier.pkl")

    # Тестируем загруженную модель
    test_result = loaded_classifier.predict_single("Тестируем загруженную модель классификации")
    print(f"   Результат предсказания: {test_result['prediction']}")


# Командный интерфейс
if __name__ == "__main__":
    print("🚀 ЗАПУСК ПРИМЕРА ИСПОЛЬЗОВАНИЯ КЛАССИФИКАТОРА")
    print("=" * 60)
    main()