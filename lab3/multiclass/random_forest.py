from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from util.jsonl_process import read_jsonl_basic


class RandomForestCategoryClassifier:
    """
    Многоклассовый классификатор категорий на основе случайного леса
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 max_features: Union[str, int, float] = 'sqrt',
                 random_state: int = 42,
                 class_names: Optional[List[str]] = None,
                 text_field: str = 'text',
                 label_field: str = 'category',
                 class_weight: Optional[str] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1):
        """
        Args:
            n_estimators: количество деревьев в лесу
            max_depth: максимальная глубина деревьев
            max_features: количество признаков для рассмотрения в каждом разбиении
            random_state: для воспроизводимости результатов
            class_names: список названий классов (опционально)
            text_field: название поля с текстом
            label_field: название поля с меткой категории
            class_weight: вес классов ('balanced', 'balanced_subsample' или None)
            min_samples_split: минимальное количество образцов для разделения узла
            min_samples_leaf: минимальное количество образцов в листе
        """
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            stop_words=None
        )

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,  # Используем все ядра процессора
            bootstrap=True,
            oob_score=True,  # Out-of-bag score для оценки качества
            class_weight=class_weight,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            verbose=0
        )

        self.label_encoder = LabelEncoder()
        self.class_names = class_names
        self.is_trained = False
        self.num_classes = 0
        self.class_mapping = {}
        self.text_field = text_field
        self.label_field = label_field
        self.random_state = random_state
        self.all_classes_fitted = []  # Сохраняем все классы, которые были в тренировочных данных

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
            'imbalance_ratio': None,
            'unique_labels': sorted(list(label_counts.keys()))
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
              auto_detect_classes: bool = True) -> None:
        """
        Обучение модели случайного леса
        """
        print("🎯 ОБУЧЕНИЕ МНОГОКЛАССОВОГО СЛУЧАЙНОГО ЛЕСА...")
        print(f"   Поле с текстом: '{self.text_field}'")
        print(f"   Поле с категорией: '{self.label_field}'")
        print(f"   Количество деревьев: {self.model.n_estimators}")
        print(f"   Максимальная глубина: {self.model.max_depth}")

        # Анализ распределения категорий
        train_dist = self.analyze_class_distribution(train_data)
        print(f"\n📊 РАСПРЕДЕЛЕНИЕ КАТЕГОРИЙ В TRAIN:")
        print(f"   Всего примеров: {train_dist['total_samples']}")
        print(f"   Количество категорий: {train_dist['num_classes']}")

        if train_dist['imbalance_ratio']:
            print(f"   Коэффициент дисбаланса: {train_dist['imbalance_ratio']:.2f}")
            if train_dist['imbalance_ratio'] > 3 and self.model.class_weight is None:
                print("   ⚠️  Обнаружен сильный дисбаланс категорий")
                print("   ℹ️  Рекомендуется использовать class_weight='balanced'")

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
        self.all_classes_fitted = self.class_names.copy()  # Сохраняем все классы из тренировочных данных

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
        print("\n🤖 Обучение случайного леса...")
        self.model.fit(X_train_vec, y_train)
        self.is_trained = True

        # Оценка на тренировочных данных
        train_pred = self.model.predict(X_train_vec)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"\n✅ Точность на train: {train_accuracy:.3f}")

        # Out-of-bag score
        if hasattr(self.model, 'oob_score_'):
            print(f"✅ Out-of-bag score: {self.model.oob_score_:.3f}")

        # Отчет по классам на train
        print("\n📊 ОТЧЕТ ПО КАТЕГОРИЯМ (train):")
        print(classification_report(y_train, train_pred, target_names=self.class_names))

        # Оценка на валидации, если есть
        if val_data:
            val_accuracy, _ = self.evaluate(val_data, detailed=False)
            print(f"✅ Точность на val: {val_accuracy:.3f}")

        # Покажем важные признаки
        self._show_important_features(top_n=20)

        # Информация о модели
        self._show_model_info()

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
        top_categories = []
        for idx in top_indices:
            top_categories.append({
                'category': self.class_names[idx],
                'probability': prob[idx],
                'probability_percent': prob[idx] * 100
            })

        return {
            'prediction': pred,
            'category': pred,
            'prediction_encoded': pred_encoded,
            'category_probabilities': class_probs,
            'top_categories': top_categories,
            'confidence': prob[pred_encoded],
            'confidence_percent': prob[pred_encoded] * 100
        }

    def _safe_transform_labels(self, labels_raw: List[str]) -> np.ndarray:
        """
        Безопасное преобразование меток, учитывая отсутствующие классы
        """
        try:
            return self.label_encoder.transform(labels_raw)
        except ValueError as e:
            # Если встречаются метки, которых нет в тренировочных данных
            print(f"⚠️  Предупреждение: в данных встретились новые категории, которых не было в тренировке")

            # Создаем маску для известных меток
            known_labels = []
            unknown_count = 0
            for label in labels_raw:
                if label in self.label_encoder.classes_:
                    known_labels.append(label)
                else:
                    known_labels.append(self.class_names[0])  # Заменяем на первый класс
                    unknown_count += 1

            if unknown_count > 0:
                print(f"   Заменено {unknown_count} неизвестных меток на '{self.class_names[0]}'")

            return self.label_encoder.transform(known_labels)

    def evaluate(self, test_data: List[Dict[str, Any]],
                 detailed: bool = True,
                 plot_confusion_matrix: bool = True) -> Tuple[float, Dict]:
        """
        Оценка модели на тестовых данных
        """
        X_test, y_test_raw = self.prepare_data(test_data)

        # Проверяем, какие категории есть в тестовых данных
        test_dist = self.analyze_class_distribution(test_data)
        test_classes = set(test_dist['unique_labels'])
        train_classes = set(self.class_names)

        missing_in_test = train_classes - test_classes
        missing_in_train = test_classes - train_classes

        if missing_in_train:
            print(f"⚠️  В тестовых данных есть категории, которых не было в тренировке: {missing_in_train}")
            print(f"   Эти категории будут проигнорированы при оценке")

        if missing_in_test:
            print(f"ℹ️  В тестовых данных отсутствуют некоторые категории из тренировки: {missing_in_test}")

        # Безопасное преобразование меток
        y_test = self._safe_transform_labels(y_test_raw)
        X_test_vec = self.vectorizer.transform(X_test)

        y_pred_encoded = self.model.predict(X_test_vec)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        accuracy = accuracy_score(y_test, y_pred_encoded)

        if detailed:
            print(f"\n📊 ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ:")
            print(f"   Примеров: {len(test_data)}")
            print(f"   Категорий в тесте: {test_dist['num_classes']}")
            print(f"   Точность: {accuracy:.3f}")

            # Получаем только те классы, которые есть и в предсказаниях и в истинных метках
            unique_true = np.unique(y_test)
            unique_pred = np.unique(y_pred_encoded)
            common_classes = np.intersect1d(unique_true, unique_pred)

            if len(common_classes) > 0:
                # Создаем маску для выбора только общих классов
                mask = np.isin(y_test, common_classes) & np.isin(y_pred_encoded, common_classes)

                if np.sum(mask) > 0:
                    y_test_filtered = y_test[mask]
                    y_pred_filtered = y_pred_encoded[mask]

                    # Получаем названия классов только для общих классов
                    available_classes = self.label_encoder.inverse_transform(common_classes)

                    print(f"\n📈 ДЕТАЛЬНЫЙ ОТЧЕТ ПО КАТЕГОРИЯМ (только общие категории):")
                    print(classification_report(
                        y_test_filtered,
                        y_pred_filtered,
                        target_names=available_classes,
                        digits=3
                    ))
                else:
                    print("❌ Нет общих категорий для оценки")
            else:
                print("❌ Нет общих категорий для оценки")

            # Матрица ошибок только для общих классов
            if len(common_classes) > 0:
                print(f"\n📊 МАТРИЦА ОШИБОК (только общие категории):")
                cm = confusion_matrix(y_test, y_pred_encoded, labels=common_classes)
                self._print_confusion_matrix_custom(cm, common_classes)

                if plot_confusion_matrix and len(common_classes) > 1:
                    self._plot_confusion_matrix_custom(
                        y_test, y_pred_encoded, common_classes,
                        "Test Confusion Matrix (Common Classes Only)"
                    )

        # Дополнительные метрики
        report_dict = {}
        try:
            # Пытаемся получить отчет только для общих классов
            unique_true = np.unique(y_test)
            unique_pred = np.unique(y_pred_encoded)
            common_classes = np.intersect1d(unique_true, unique_pred)

            if len(common_classes) > 0:
                mask = np.isin(y_test, common_classes) & np.isin(y_pred_encoded, common_classes)
                if np.sum(mask) > 0:
                    y_test_filtered = y_test[mask]
                    y_pred_filtered = y_pred_encoded[mask]
                    available_classes = self.label_encoder.inverse_transform(common_classes)

                    report_dict = classification_report(
                        y_test_filtered,
                        y_pred_filtered,
                        target_names=available_classes,
                        output_dict=True
                    )
        except Exception as e:
            print(f"⚠️  Не удалось получить детальный отчет: {e}")

        return accuracy, report_dict

    def _print_confusion_matrix_custom(self, cm: np.ndarray, classes: np.ndarray) -> None:
        """
        Печать матрицы ошибок для заданных классов
        """
        n_classes = len(classes)
        class_names = self.label_encoder.inverse_transform(classes)

        if n_classes == 0:
            print("❌ Нет классов для отображения матрицы ошибок")
            return

        # Заголовок
        max_class_len = max(len(cls) for cls in class_names)
        header_padding = max(12, max_class_len + 2)

        header = " " * header_padding + " | "
        header += " ".join([f"{cls[:10]:>10}" for cls in class_names])
        print(header)
        print("-" * (header_padding + 3 + n_classes * 11))

        # Строки
        for i, cls in enumerate(class_names):
            row = f"{cls[:header_padding - 2]:>{header_padding - 2}} | "
            row += " ".join([f"{cm[i][j]:>10}" for j in range(n_classes)])
            print(row)

    def _plot_confusion_matrix_custom(self, y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      classes: np.ndarray,
                                      title: str = "Confusion Matrix") -> None:
        """
        Визуализация матрицы ошибок для заданных классов
        """
        try:
            if len(classes) <= 1:
                print("⚠️  Недостаточно классов для построения матрицы ошибок")
                return

            cm = confusion_matrix(y_true, y_pred, labels=classes)
            class_names = self.label_encoder.inverse_transform(classes)

            plt.figure(figsize=(max(8, len(classes)), max(6, len(classes) * 0.7)))

            # Нормализуем по строкам (по истинным меткам)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)  # Заменяем NaN на 0

            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=class_names,
                        yticklabels=class_names,
                        vmin=0, vmax=1)
            plt.title(f"{title} (нормализована)")
            plt.ylabel('Истинные категории')
            plt.xlabel('Предсказанные категории')
            plt.tight_layout()

            filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '')
            plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"⚠️  Не удалось построить матрицу ошибок: {e}")

    def _plot_confusion_matrix(self, y_true: np.ndarray,
                               y_pred: np.ndarray,
                               title: str = "Confusion Matrix") -> None:
        """
        Визуализация матрицы ошибок для всех классов
        """
        try:
            if self.num_classes <= 1:
                print("⚠️  Недостаточно классов для построения матрицы ошибок")
                return

            cm = confusion_matrix(y_true, y_pred)

            plt.figure(figsize=(max(10, self.num_classes), max(8, self.num_classes * 0.8)))

            # Нормализуем по строкам (по истинным меткам)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)  # Заменяем NaN на 0

            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=self.class_names,
                        yticklabels=self.class_names,
                        vmin=0, vmax=1)
            plt.title(f"{title} (нормализована)")
            plt.ylabel('Истинные категории')
            plt.xlabel('Предсказанные категории')
            plt.tight_layout()

            filename = title.lower().replace(' ', '_').replace('-', '_')
            plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"⚠️  Не удалось построить матрицу ошибок: {e}")

    def _show_important_features(self, top_n: int = 20):
        """
        Показывает самые важные признаки
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("❌ Не удалось получить важность признаков")
            return

        feature_names = self.vectorizer.get_feature_names_out()
        importances = self.model.feature_importances_

        print(f"\n🔍 ТОП-{top_n} ВАЖНЫХ ПРИЗНАКОВ (Random Forest):")

        # Сортируем признаки по важности
        indices = np.argsort(importances)[::-1]

        print(f"\n   САМЫЕ ВАЖНЫЕ ПРИЗНАКИ ДЛЯ ВСЕХ КАТЕГОРИЙ:")
        for i in range(min(top_n, len(indices))):
            idx = indices[i]
            print(f"      {i + 1:2d}. {feature_names[idx]:20s}: {importances[idx]:.5f}")

        # Анализ важности по категориям (косвенный)
        print(f"\n   📊 ОБЩАЯ ИНФОРМАЦИЯ:")
        total_importance = np.sum(importances)
        top_n_importance = np.sum(importances[indices[:top_n]])
        print(f"      Топ-{top_n} признаков объясняют {top_n_importance / total_importance * 100:.1f}% общей важности")

        # Доля признаков с нулевой важностью
        zero_importance_count = np.sum(importances == 0)
        print(
            f"      Признаков с нулевой важностью: {zero_importance_count} ({zero_importance_count / len(importances) * 100:.1f}%)")

    def _show_model_info(self):
        """
        Показывает информацию о обученной модели
        """
        print(f"\n📊 ИНФОРМАЦИЯ О СЛУЧАЙНОМ ЛЕСЕ:")
        print(f"   Количество деревьев: {len(self.model.estimators_)}")

        # Глубина деревьев
        depths = [est.tree_.max_depth for est in self.model.estimators_]
        if depths:
            print(f"   Глубина деревьев: {np.min(depths)} (мин), {np.mean(depths):.1f} (ср), {np.max(depths)} (макс)")

        # Среднее количество листьев
        n_leaves = [est.tree_.n_leaves for est in self.model.estimators_]
        if n_leaves:
            print(f"   Листья в дереве: {np.mean(n_leaves):.0f} (в среднем)")

        if hasattr(self.model, 'oob_score_'):
            print(f"   Out-of-bag score: {self.model.oob_score_:.3f}")

    def get_feature_importance_df(self, top_n: int = 50) -> Optional[pd.DataFrame]:
        """
        Возвращает DataFrame с важностью признаков
        """
        if not hasattr(self.model, 'feature_importances_'):
            return None

        feature_names = self.vectorizer.get_feature_names_out()
        importances = self.model.feature_importances_

        # Сортируем по важности
        indices = np.argsort(importances)[::-1]

        data = {
            'feature': feature_names[indices[:top_n]],
            'importance': importances[indices[:top_n]],
            'rank': range(1, top_n + 1)
        }

        return pd.DataFrame(data)

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
            'all_classes_fitted': self.all_classes_fitted,
            'text_field': self.text_field,
            'label_field': self.label_field,
            'random_state': self.random_state
        }, filename)
        print(f"💾 Модель случайного леса сохранена: {filename}")

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
        self.all_classes_fitted = loaded.get('all_classes_fitted', self.class_names)
        self.text_field = loaded.get('text_field', 'text')
        self.label_field = loaded.get('label_field', 'category')
        self.random_state = loaded.get('random_state', 42)
        self.is_trained = True

        print(f"📥 Модель случайного леса загружена: {filename}")
        print(f"   Категории: {self.class_names}")
        print(f"   Количество категорий: {self.num_classes}")
        print(f"   Количество деревьев: {self.model.n_estimators}")

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
                'top_categories': top_categories
            })

        return results


def compare_rf_parameters(train_data: List[Dict[str, Any]],
                          val_data: Optional[List[Dict[str, Any]]] = None,
                          text_field: str = 'text',
                          label_field: str = 'category',
                          use_subset: bool = True) -> Dict[str, RandomForestCategoryClassifier]:
    """
    Сравнение разных параметров случайного леса (безопасная версия)
    """
    print("🔬 СРАВНЕНИЕ ПАРАМЕТРОВ СЛУЧАЙНОГО ЛЕСА")
    print("=" * 60)

    models = {}

    # Используем подмножество для скорости, если нужно
    if use_subset and len(train_data) > 300:
        train_subset = train_data[:300]
        print(f"ℹ️  Используем подмножество из {len(train_subset)} примеров для сравнения")
    else:
        train_subset = train_data

    if use_subset and val_data and len(val_data) > 100:
        val_subset = val_data[:100]
        print(f"ℹ️  Используем подмножество из {len(val_subset)} примеров для валидации")
    else:
        val_subset = val_data

    # Определяем классы из тренировочных данных
    all_labels = [item[label_field] for item in train_subset]
    class_names = sorted(list(set(all_labels)))

    if len(class_names) < 2:
        print("❌ Недостаточно категорий для сравнения (минимум 2)")
        return models

    # 1. Разное количество деревьев (только 2 варианта для скорости)
    print("\n1. СРАВНЕНИЕ РАЗНОГО КОЛИЧЕСТВА ДЕРЕВЬЕВ:")
    for n_trees in [50, 100]:  # Уменьшили количество вариантов
        print(f"\n   Random Forest с {n_trees} деревьями:")
        try:
            model = RandomForestCategoryClassifier(
                n_estimators=n_trees,
                max_depth=None,
                class_names=class_names,
                text_field=text_field,
                label_field=label_field,
                class_weight=None
            )
            model.train(train_subset, val_subset, auto_detect_classes=False)
            models[f'RF_{n_trees}_trees'] = model
        except Exception as e:
            print(f"   ❌ Ошибка при обучении с {n_trees} деревьями: {e}")

    # 2. Разная максимальная глубина (только 2 варианта)
    print("\n2. СРАВНЕНИЕ РАЗНОЙ ГЛУБИНЫ ДЕРЕВЬЕВ:")
    for depth in [10, None]:  # Уменьшили количество вариантов
        depth_name = "None" if depth is None else depth
        print(f"\n   Random Forest с max_depth={depth_name}:")
        try:
            model = RandomForestCategoryClassifier(
                n_estimators=50,  # Меньше деревьев для скорости
                max_depth=depth,
                class_names=class_names,
                text_field=text_field,
                label_field=label_field,
                class_weight=None
            )
            model.train(train_subset, val_subset, auto_detect_classes=False)
            models[f'RF_depth_{depth_name}'] = model
        except Exception as e:
            print(f"   ❌ Ошибка при обучении с max_depth={depth_name}: {e}")

    return models


def analyze_feature_importance(model: RandomForestCategoryClassifier,
                               top_n: int = 30,
                               save_to_csv: bool = True) -> None:
    """
    Детальный анализ важности признаков
    """
    importance_df = model.get_feature_importance_df(top_n=top_n)

    if importance_df is not None:
        print(f"\n📈 ДЕТАЛЬНЫЙ АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ (Топ-{top_n}):")
        print("=" * 60)

        # Выводим таблицу
        pd.set_option('display.max_rows', top_n)
        print(importance_df.to_string(index=False))
        pd.reset_option('display.max_rows')

        # Сохраняем в CSV
        if save_to_csv:
            csv_filename = f"feature_importance_{len(model.class_names)}_categories.csv"
            importance_df.to_csv(csv_filename, index=False)
            print(f"\n💾 Важность признаков сохранена в: {csv_filename}")

        # Визуализация
        try:
            plt.figure(figsize=(12, 8))
            plt.barh(range(top_n), importance_df['importance'].values[:top_n][::-1])
            plt.yticks(range(top_n), importance_df['feature'].values[:top_n][::-1])
            plt.xlabel('Важность признака')
            plt.title(f'Топ-{top_n} важнейших признаков для {len(model.class_names)} категорий')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"⚠️  Не удалось построить график важности признаков: {e}")


def quick_train_rf(train_file: str,
                   val_file: Optional[str] = None,
                   test_file: Optional[str] = None,
                   text_field: str = 'text',
                   label_field: str = 'category',
                   n_estimators: int = 100,
                   max_depth: Optional[int] = None,
                   class_weight: Optional[str] = None,
                   output_model: str = 'rf_category_classifier.pkl',
                   use_subset_for_training: bool = False) -> Optional[RandomForestCategoryClassifier]:
    """
    Быстрое обучение случайного леса из файлов
    """
    import json
    import os

    def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
        if not os.path.exists(filepath):
            print(f"⚠️  Файл не найден: {filepath}")
            return []
        with open(filepath, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    print("🚀 ЗАПУСК БЫСТРОГО ОБУЧЕНИЯ СЛУЧАЙНОГО ЛЕСА")
    print("=" * 60)

    # Загрузка данных
    print(f"\n📥 Загрузка данных...")
    train_data = load_jsonl(train_file)
    if not train_data:
        print(f"❌ Ошибка: не удалось загрузить тренировочные данные из {train_file}")
        return None

    print(f"   Train: {len(train_data)} примеров")

    # Используем подмножество для обучения, если нужно
    if use_subset_for_training and len(train_data) > 1000:
        train_data = train_data[:1000]
        print(f"   ℹ️  Используем подмножество из {len(train_data)} примеров для обучения")

    if val_file:
        val_data = load_jsonl(val_file)
        print(f"   Val: {len(val_data)} примеров")
    else:
        val_data = None

    if test_file:
        test_data = load_jsonl(test_file)
        print(f"   Test: {len(test_data)} примеров")
    else:
        test_data = None

    # Проверяем структуру данных
    if train_data:
        sample_item = train_data[0]
        if text_field not in sample_item:
            print(f"❌ Ошибка: поле '{text_field}' не найдено в данных")
            return None
        if label_field not in sample_item:
            print(f"❌ Ошибка: поле '{label_field}' не найдено в данных")
            return None

    # Обучаем модель
    print(f"\n🎯 Начало обучения случайного леса...")
    print(f"   Параметры: {n_estimators} деревьев, max_depth={max_depth}")

    classifier = RandomForestCategoryClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        text_field=text_field,
        label_field=label_field
    )

    classifier.train(train_data, val_data, auto_detect_classes=True)

    # Тестируем, если есть тестовые данные
    if test_data:
        print(f"\n🧪 Тестирование на тестовых данных...")
        accuracy, report = classifier.evaluate(test_data, detailed=True)
        print(f"\n🎯 Итоговая точность на тесте: {accuracy:.3f}")

        # Сохраняем отчет
        if report:
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv('rf_classification_report.csv', index=True)
            print(f"📄 Детальный отчет сохранен в 'rf_classification_report.csv'")

    # Анализ важности признаков
    analyze_feature_importance(classifier, top_n=25, save_to_csv=True)

    # Сохраняем модель
    classifier.save_model(output_model)

    # Тестовый пример
    print(f"\n🧪 ТЕСТОВЫЙ ПРИМЕР РАБОТЫ МОДЕЛИ:")
    if train_data:
        sample_text = train_data[0][text_field]
        if len(sample_text) > 100:
            sample_text = sample_text[:100] + "..."
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
    Пример использования случайного леса для классификации категорий
    """
    try:
        # Загрузка данных
        train_data = read_jsonl_basic('../../util/news_category_train.jsonl')
        val_data = read_jsonl_basic('../../util/news_category_val.jsonl')
        test_data = read_jsonl_basic('../../util/news_category_test.jsonl')

        print(f"📊 Данные: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

        # Проверяем структуру данных
        if train_data:
            print(f"\n📋 ПРИМЕР ДАННЫХ:")
            sample = train_data[0]
            print(f"   Поля: {list(sample.keys())}")
            print(f"   Текст: {sample.get('text', 'N/A')[:100]}...")
            print(f"   Категория: {sample.get('category', 'N/A')}")

        # 1. Обучаем базовую модель случайного леса
        print("\n" + "=" * 60)

        rf_classifier = RandomForestCategoryClassifier(
            n_estimators=50,  # Меньше деревьев для скорости
            max_depth=None,
            text_field='text',
            label_field='category',
            class_weight=None
        )

        # Используем подмножество для демонстрации, если данных много
        if len(train_data) > 500:
            print(f"ℹ️  Используем подмножество из 500 примеров для демонстрации")
            train_subset = train_data[:500]
            val_subset = val_data[:100] if val_data and len(val_data) > 100 else val_data
        else:
            train_subset = train_data
            val_subset = val_data

        rf_classifier.train(train_subset, val_subset, auto_detect_classes=True)

        # 3. Оценка на тестовых данных (если есть)
        if test_data and len(test_data) > 0:
            print("\n" + "=" * 60)
            print("🧪 ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ:")
            test_subset = test_data[:200] if len(test_data) > 200 else test_data
            test_accuracy, test_report = rf_classifier.evaluate(test_subset, detailed=True)
            print(f"\n📊 Итоговая точность на тесте: {test_accuracy:.3f}")

        # 4. Детальный анализ важности признаков
        print("\n" + "=" * 60)
        analyze_feature_importance(rf_classifier, top_n=15, save_to_csv=True)

        # 5. Сохраняем модель
        rf_classifier.save_model("random_forest_category_classifier.pkl")

        # 6. Сравниваем разные параметры (на небольшом подмножестве для скорости)
        print("\n" + "=" * 60)
        print("🔬 СРАВНЕНИЕ ПАРАМЕТРОВ (на подмножестве):")

        small_train = train_data[:200] if len(train_data) > 200 else train_data
        small_val = val_data[:50] if val_data and len(val_data) > 50 else val_data

        if small_train and small_val:
            models = compare_rf_parameters(
                small_train, small_val,
                'text', 'category',
                use_subset=False  # Уже используем подмножество
            )
            print(f"\n✅ Сравнение завершено. Обучено моделей: {len(models)}")
        else:
            print("❌ Недостаточно данных для сравнения параметров")

    except FileNotFoundError as e:
        print(f"❌ Файл не найден: {e}")
        print("ℹ️  Проверьте пути к файлам данных")
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Запуск примера
    print("🚀 ЗАПУСК ПРИМЕРА ИСПОЛЬЗОВАНИЯ СЛУЧАЙНОГО ЛЕСА")
    print("=" * 60)
    main()