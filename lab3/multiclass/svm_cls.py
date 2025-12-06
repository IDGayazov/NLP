from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
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


class SVMCategoryClassifier:
    """
    Многоклассовый классификатор категорий на основе SVM
    Поддерживает линейные и RBF ядра
    """

    def __init__(self,
                 C: float = 1.0,
                 kernel: str = 'linear',
                 loss: str = 'squared_hinge',
                 penalty: str = 'l2',
                 class_names: Optional[List[str]] = None,
                 text_field: str = 'text',
                 label_field: str = 'category',
                 calibrate_probabilities: bool = True,
                 multi_class_strategy: str = 'ovr',
                 random_state: int = 42):
        """
        Args:
            C: параметр регуляризации (меньше = сильнее регуляризация)
            kernel: тип ядра ('linear', 'rbf', 'poly', 'sigmoid')
            loss: функция потерь ('hinge' или 'squared_hinge') - только для linear
            penalty: тип регуляризации ('l1' или 'l2') - только для linear
            class_names: список названий классов (опционально)
            text_field: название поля с текстом
            label_field: название поля с меткой категории
            calibrate_probabilities: калибровать вероятности (рекомендуется для SVM)
            multi_class_strategy: стратегия многоклассовой классификации ('ovr' или 'ovr')
            random_state: для воспроизводимости
        """
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            stop_words=None
        )

        # Выбираем тип SVM в зависимости от ядра
        if kernel == 'linear':
            base_svm = LinearSVC(
                C=C,
                loss=loss,
                penalty=penalty,
                dual=True,
                random_state=random_state,
                max_iter=2000,
                multi_class=multi_class_strategy
            )
        else:
            # Для нелинейных ядер используем SVC
            base_svm = SVC(
                C=C,
                kernel=kernel,
                random_state=random_state,
                max_iter=2000,
                decision_function_shape=multi_class_strategy,
                probability=calibrate_probabilities  # Включаем вероятности если калибровка
            )

        # Калибровка вероятностей для SVM (только для linear или если probability=False в SVC)
        if calibrate_probabilities:
            if kernel == 'linear' or not base_svm.probability:
                self.model = CalibratedClassifierCV(base_svm, cv=3, method='sigmoid')
            else:
                self.model = base_svm  # SVC уже имеет вероятности при probability=True
        else:
            self.model = base_svm

        self.label_encoder = LabelEncoder()
        self.class_names = class_names
        self.text_field = text_field
        self.label_field = label_field
        self.calibrate_probabilities = calibrate_probabilities
        self.kernel = kernel
        self.C = C
        self.multi_class_strategy = multi_class_strategy
        self.is_trained = False
        self.num_classes = 0
        self.random_state = random_state

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
        Обучение модели SVM
        """
        print("🎯 ОБУЧЕНИЕ SVM ДЛЯ МНОГОКЛАССОВОЙ КЛАССИФИКАЦИИ...")
        print(f"   Поле с текстом: '{self.text_field}'")
        print(f"   Поле с категорией: '{self.label_field}'")
        print(f"   Ядро: {self.kernel}")
        print(f"   Параметр C: {self.C}")
        print(f"   Калибровка вероятностей: {self.calibrate_probabilities}")
        print(f"   Стратегия многоклассовой классификации: {self.multi_class_strategy}")

        # Анализ распределения категорий
        train_dist = self.analyze_class_distribution(train_data)
        print(f"\n📊 РАСПРЕДЕЛЕНИЕ КАТЕГОРИЙ В TRAIN:")
        print(f"   Всего примеров: {train_dist['total_samples']}")
        print(f"   Количество категорий: {train_dist['num_classes']}")

        if train_dist['imbalance_ratio']:
            print(f"   Коэффициент дисбаланса: {train_dist['imbalance_ratio']:.2f}")
            if train_dist['imbalance_ratio'] > 3:
                print("   ⚠️  Обнаружен сильный дисбаланс категорий")
                print("   ℹ️  Для SVM рекомендуется использовать class_weight='balanced'")

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
        print("\n🤖 Обучение SVM...")
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

        # Покажем важные признаки (только для линейного ядра)
        if self.kernel == 'linear':
            self._show_important_features(top_n=15)

        # Информация о поддержке векторов
        self._show_svm_info(X_train_vec)

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

        # Получаем вероятности
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_vec)
        else:
            # Если вероятности не доступны, используем decision function
            decision_scores = self.model.decision_function(X_vec)
            probabilities = self._decision_to_probability_multiclass(decision_scores)

        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        return predictions, probabilities

    def _decision_to_probability_multiclass(self, decision_scores: np.ndarray) -> np.ndarray:
        """
        Преобразование decision function в вероятности для многоклассовой классификации
        """
        # Для многоклассового случая decision_scores имеет размерность (n_samples, n_classes)
        if len(decision_scores.shape) == 1:
            # Бинарный случай
            decision_scores = decision_scores.reshape(-1, 1)
            decision_scores = np.hstack([-decision_scores, decision_scores])

        # Используем softmax для преобразования в вероятности
        exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probabilities

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

        # Decision function для получения уверенности
        X_vec = self.vectorizer.transform([text])
        if hasattr(self.model, 'decision_function'):
            decision_scores = self.model.decision_function(X_vec)[0]
            # Берем расстояние до гиперплоскости для предсказанного класса
            if len(decision_scores.shape) == 0:
                decision_score = abs(decision_scores)
            else:
                decision_score = abs(decision_scores[pred_encoded])
        else:
            # Если decision function недоступен, используем разность вероятностей
            decision_score = prob[pred_encoded] - np.max(prob[np.arange(len(prob)) != pred_encoded])

        return {
            'prediction': pred,
            'category': pred,
            'prediction_encoded': pred_encoded,
            'category_probabilities': class_probs,
            'top_categories': top_categories,
            'confidence': prob[pred_encoded],
            'confidence_percent': prob[pred_encoded] * 100,
            'decision_score': decision_score,
            'margin': self._calculate_margin(X_vec, pred_encoded)
        }

    def _calculate_margin(self, X_vec, pred_encoded):
        """
        Рассчитывает зазор (margin) для предсказания
        """
        if hasattr(self.model, 'decision_function'):
            decision_scores = self.model.decision_function(X_vec)[0]

            if len(decision_scores.shape) == 0:
                # Бинарный случай
                return abs(decision_scores)
            else:
                # Многоклассовый случай
                scores = decision_scores.copy()
                scores[pred_encoded] = -np.inf  # Исключаем предсказанный класс
                second_best = np.max(scores)
                margin = decision_scores[pred_encoded] - second_best
                return margin
        return 0.0

    def evaluate(self, test_data: List[Dict[str, Any]],
                 detailed: bool = True,
                 plot_confusion_matrix: bool = True) -> Tuple[float, Dict]:
        """
        Оценка модели на тестовых данных
        """
        X_test, y_test_raw = self.prepare_data(test_data)

        # Безопасное преобразование меток
        y_test = []
        for label in y_test_raw:
            if label in self.label_encoder.classes_:
                y_test.append(label)
            else:
                # Заменяем неизвестные метки на первую известную
                y_test.append(self.class_names[0])

        y_test_encoded = self.label_encoder.transform(y_test)
        X_test_vec = self.vectorizer.transform(X_test)

        y_pred_encoded = self.model.predict(X_test_vec)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)

        if detailed:
            print(f"\n📊 ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ:")
            print(f"   Примеров: {len(test_data)}")
            print(f"   Точность: {accuracy:.3f}")

            print(f"\n📈 ДЕТАЛЬНЫЙ ОТЧЕТ ПО КАТЕГОРИЯМ:")
            print(classification_report(y_test_encoded, y_pred_encoded,
                                        target_names=self.class_names, digits=3))

            # Матрица ошибок
            if self.num_classes > 1:
                print(f"\n📊 МАТРИЦА ОШИБОК:")
                cm = confusion_matrix(y_test_encoded, y_pred_encoded)
                self._print_confusion_matrix(cm)

                if plot_confusion_matrix:
                    self._plot_confusion_matrix(y_test_encoded, y_pred_encoded,
                                                "Test Confusion Matrix")

        # Дополнительные метрики
        report_dict = classification_report(y_test_encoded, y_pred_encoded,
                                            target_names=self.class_names,
                                            output_dict=True)

        return accuracy, report_dict

    def _print_confusion_matrix(self, cm: np.ndarray) -> None:
        """
        Красиво печатает матрицу ошибок
        """
        n_classes = len(self.class_names)

        if n_classes <= 1:
            print("❌ Недостаточно классов для матрицы ошибок")
            return

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

    def _show_important_features(self, top_n: int = 15):
        """
        Показывает самые важные признаки для каждого класса (только для линейного ядра)
        """
        # Получаем коэффициенты из базового estimator
        coef = None

        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_
        elif hasattr(self.model, 'estimators_'):
            # Для CalibratedClassifierCV берем первый estimator
            for estimator in self.model.estimators_:
                if hasattr(estimator, 'coef_'):
                    coef = estimator.coef_
                    break

        if coef is None:
            print("ℹ️  Важность признаков доступна только для линейного ядра")
            return

        feature_names = self.vectorizer.get_feature_names_out()

        print(f"\n🔍 ТОП-{top_n} ВАЖНЫХ ПРИЗНАКОВ SVM (линейное ядро):")

        # Для каждого класса показываем свои важные признаки
        for class_idx, class_name in enumerate(self.class_names):
            print(f"\n   КАТЕГОРИЯ '{class_name}':")

            if len(coef.shape) == 1:
                # Бинарный случай
                coef_for_class = coef
            else:
                # Многоклассовый случай
                coef_for_class = coef[class_idx]

            # Положительные признаки (указывают на данную категорию)
            print(f"      Показатели ДЛЯ категории:")
            pos_indices = np.argsort(coef_for_class)[-top_n:][::-1]
            for idx in pos_indices:
                print(f"        + {feature_names[idx]}: {coef_for_class[idx]:.3f}")

            # Отрицательные признаки (указывают ПРОТИВ данной категории)
            print(f"      Показатели ПРОТИВ категории:")
            neg_indices = np.argsort(coef_for_class)[:top_n]
            for idx in neg_indices:
                print(f"        - {feature_names[idx]}: {coef_for_class[idx]:.3f}")

    def _show_svm_info(self, X_train_vec: Any) -> None:
        """
        Показывает информацию о векторах поддержки
        """
        try:
            # Получаем базовый estimator
            base_estimator = None
            if hasattr(self.model, 'estimators_'):
                # Для калиброванной модели
                base_estimator = self.model.estimators_[0]
            else:
                base_estimator = self.model

            if hasattr(base_estimator, 'support_'):
                n_support_vectors = len(base_estimator.support_)
                print(f"\n📊 ИНФОРМАЦИЯ О SVM:")
                print(f"   Количество векторов поддержки: {n_support_vectors}")
                print(f"   Процент от обучающей выборки: {n_support_vectors / X_train_vec.shape[0] * 100:.1f}%")

                if hasattr(base_estimator, 'support_vectors_'):
                    print(f"   Размерность векторов поддержки: {base_estimator.support_vectors_.shape}")

        except Exception as e:
            print(f"   ℹ️  Информация о векторах поддержки недоступна: {e}")

    def get_decision_boundary_info(self, text: str) -> Dict[str, Any]:
        """
        Получить информацию о расстоянии до разделяющей гиперплоскости
        """
        X_vec = self.vectorizer.transform([text])

        if hasattr(self.model, 'decision_function'):
            decision_scores = self.model.decision_function(X_vec)[0]

            if len(decision_scores.shape) == 0:
                # Бинарный случай
                decision_score = decision_scores
                distance_from_boundary = abs(decision_score)
                side = "positive" if decision_score > 0 else "negative"
            else:
                # Многоклассовый случай
                decision_score = np.max(decision_scores)
                predicted_class = np.argmax(decision_scores)
                distance_from_boundary = decision_score
                side = self.class_names[predicted_class]
        else:
            # Если decision function недоступен
            decision_scores = None
            distance_from_boundary = 0
            side = "unknown"

        return {
            'decision_scores': decision_scores,
            'distance_from_boundary': distance_from_boundary,
            'side': side,
            'confidence': min(distance_from_boundary, 1.0) if distance_from_boundary is not None else 0.5
        }

    def save_model(self, filename: str) -> None:
        """
        Сохранение модели
        """
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'class_names': self.class_names,
            'text_field': self.text_field,
            'label_field': self.label_field,
            'calibrate_probabilities': self.calibrate_probabilities,
            'kernel': self.kernel,
            'C': self.C,
            'multi_class_strategy': self.multi_class_strategy,
            'num_classes': self.num_classes
        }, filename)
        print(f"💾 Модель SVM сохранена: {filename}")

    def load_model(self, filename: str) -> None:
        """
        Загрузка модели
        """
        loaded = joblib.load(filename)
        self.model = loaded['model']
        self.vectorizer = loaded['vectorizer']
        self.label_encoder = loaded['label_encoder']
        self.class_names = loaded['class_names']
        self.text_field = loaded.get('text_field', 'text')
        self.label_field = loaded.get('label_field', 'category')
        self.calibrate_probabilities = loaded.get('calibrate_probabilities', True)
        self.kernel = loaded.get('kernel', 'linear')
        self.C = loaded.get('C', 1.0)
        self.multi_class_strategy = loaded.get('multi_class_strategy', 'ovr')
        self.num_classes = loaded.get('num_classes', len(self.class_names))
        self.is_trained = True

        print(f"📥 Модель SVM загружена: {filename}")
        print(f"   Категории: {self.class_names}")
        print(f"   Количество категорий: {self.num_classes}")
        print(f"   Ядро: {self.kernel}")
        print(f"   Параметр C: {self.C}")

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


def compare_svm_kernels(train_data: List[Dict[str, Any]],
                        val_data: List[Dict[str, Any]],
                        text_field: str = 'text',
                        label_field: str = 'category') -> Dict[str, SVMCategoryClassifier]:
    """
    Сравнение разных ядер SVM
    """
    print("🔬 СРАВНЕНИЕ РАЗНЫХ ЯДЕР SVM")
    print("=" * 60)

    kernels = ['linear', 'rbf']  # Можно добавить 'poly', 'sigmoid' если нужно
    models = {}

    # Определяем классы из данных
    all_labels = [item[label_field] for item in train_data]
    class_names = sorted(list(set(all_labels)))

    if len(class_names) < 2:
        print("❌ Недостаточно категорий для сравнения (минимум 2)")
        return models

    for kernel in kernels:
        print(f"\n🎯 SVM с ядром '{kernel}':")
        try:
            model = SVMCategoryClassifier(
                C=1.0,
                kernel=kernel,
                class_names=class_names,
                text_field=text_field,
                label_field=label_field,
                calibrate_probabilities=True
            )

            # Используем подмножество для скорости при сравнении
            if len(train_data) > 300:
                train_subset = train_data[:300]
                val_subset = val_data[:100] if len(val_data) > 100 else val_data
                print(f"   Используем подмножество: {len(train_subset)} train, {len(val_subset)} val")
            else:
                train_subset = train_data
                val_subset = val_data

            model.train(train_subset, val_subset, auto_detect_classes=False)
            models[f'SVM_{kernel}'] = model

        except Exception as e:
            print(f"   ❌ Ошибка при обучении с ядром '{kernel}': {e}")

    return models


def compare_svm_C_values(train_data: List[Dict[str, Any]],
                         val_data: List[Dict[str, Any]],
                         text_field: str = 'text',
                         label_field: str = 'category') -> Dict[str, SVMCategoryClassifier]:
    """
    Сравнение разных значений параметра C
    """
    print("🔬 СРАВНЕНИЕ РАЗНЫХ ЗНАЧЕНИЙ ПАРАМЕТРА C")
    print("=" * 60)

    C_values = [0.1, 1.0, 10.0]
    models = {}

    # Определяем классы из данных
    all_labels = [item[label_field] for item in train_data]
    class_names = sorted(list(set(all_labels)))

    if len(class_names) < 2:
        print("❌ Недостаточно категорий для сравнения (минимум 2)")
        return models

    for C in C_values:
        print(f"\n🎯 SVM с C={C}:")
        try:
            model = SVMCategoryClassifier(
                C=C,
                kernel='linear',
                class_names=class_names,
                text_field=text_field,
                label_field=label_field,
                calibrate_probabilities=True
            )

            # Используем подмножество для скорости
            if len(train_data) > 200:
                train_subset = train_data[:200]
                val_subset = val_data[:50] if len(val_data) > 50 else val_data
            else:
                train_subset = train_data
                val_subset = val_data

            model.train(train_subset, val_subset, auto_detect_classes=False)
            models[f'SVM_C_{C}'] = model

        except Exception as e:
            print(f"   ❌ Ошибка при обучении с C={C}: {e}")

    return models


def quick_train_svm(train_file: str,
                    val_file: Optional[str] = None,
                    test_file: Optional[str] = None,
                    text_field: str = 'text',
                    label_field: str = 'category',
                    kernel: str = 'linear',
                    C: float = 1.0,
                    output_model: str = 'svm_category_classifier.pkl') -> Optional[SVMCategoryClassifier]:
    """
    Быстрое обучение SVM модели из файлов
    """
    import json
    import os

    def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
        if not os.path.exists(filepath):
            print(f"⚠️  Файл не найден: {filepath}")
            return []
        with open(filepath, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    print("🚀 ЗАПУСК БЫСТРОГО ОБУЧЕНИЯ SVM")
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
    print(f"\n🎯 Начало обучения SVM...")
    print(f"   Ядро: {kernel}")
    print(f"   Параметр C: {C}")

    classifier = SVMCategoryClassifier(
        C=C,
        kernel=kernel,
        text_field=text_field,
        label_field=label_field,
        calibrate_probabilities=True
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
            report_df.to_csv('svm_classification_report.csv', index=True)
            print(f"📄 Детальный отчет сохранен в 'svm_classification_report.csv'")

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
    Пример использования SVM классификатора для многоклассовой классификации
    """
    try:
        # Загрузка данных
        train_data = read_jsonl_basic('../util/news_category_train.jsonl')
        val_data = read_jsonl_basic('../util/news_category_val.jsonl')
        test_data = read_jsonl_basic('../util/news_category_test.jsonl')

        print(f"📊 Данные: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

        # Проверяем структуру данных
        if train_data:
            print(f"\n📋 ПРИМЕР ДАННЫХ:")
            sample = train_data[0]
            print(f"   Поля: {list(sample.keys())}")
            print(f"   Текст: {sample.get('text', 'N/A')[:100]}...")
            print(f"   Категория: {sample.get('category', 'N/A')}")

        # 1. Обучаем базовую модель SVM с линейным ядром
        print("\n" + "=" * 60)

        svm_classifier = SVMCategoryClassifier(
            C=1.0,
            kernel='linear',
            text_field='text',
            label_field='category',
            calibrate_probabilities=True
        )

        svm_classifier.train(train_data, val_data, auto_detect_classes=True)

        # 3. Оценка на тестовых данных (если есть)
        if test_data and len(test_data) > 0:
            print("\n" + "=" * 60)
            print("🧪 ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ:")
            test_subset = test_data[:200] if len(test_data) > 200 else test_data
            test_accuracy, test_report = svm_classifier.evaluate(test_subset, detailed=True)
            print(f"\n📊 Итоговая точность на тесте: {test_accuracy:.3f}")

        # 4. Сохраняем модель
        svm_classifier.save_model("svm_category_classifier.pkl")
    except FileNotFoundError as e:
        print(f"❌ Файл не найден: {e}")
        print("ℹ️  Проверьте пути к файлам данных")
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Запуск примера
    print("🚀 ЗАПУСК ПРИМЕРА ИСПОЛЬЗОВАНИЯ SVM")
    print("=" * 60)
    main()