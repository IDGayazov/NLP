from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter
import pandas as pd

# Для CatBoost (опционально)
try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  CatBoost не установлен. Используем только sklearn модели.")
    CATBOOST_AVAILABLE = False

from util.jsonl_process import read_jsonl_basic

warnings.filterwarnings('ignore')


class StackingCategoryClassifier:
    """
    Стекинг/блендинг классификатор категорий для многоклассовой классификации
    """

    def __init__(self,
                 use_blending: bool = True,
                 meta_model: str = 'logistic',
                 class_names: Optional[List[str]] = None,
                 text_field: str = 'text',
                 label_field: str = 'category',
                 random_state: int = 42,
                 use_catboost: bool = False):
        """
        Args:
            use_blending: True для блендинга, False для стекинга
            meta_model: тип мета-модели ('logistic', 'svm', 'random_forest')
            class_names: список названий классов (опционально)
            text_field: название поля с текстом
            label_field: название поля с меткой категории
            random_state: для воспроизводимости
            use_catboost: использовать CatBoost в ансамбле
        """
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            stop_words=None
        )

        # Базовые модели (level-0)
        self.base_models = {
            'svm': LinearSVC(
                C=1.0,
                random_state=random_state,
                max_iter=2000,
                dual=True,
                multi_class='ovr'
            ),
            'logistic': LogisticRegression(
                C=1.0,
                random_state=random_state,
                max_iter=1000,
                solver='lbfgs',
                multi_class='multinomial'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                max_depth=None,
                n_jobs=-1
            )
        }

        # Добавляем CatBoost если доступен и запрошен
        if use_catboost and CATBOOST_AVAILABLE:
            self.base_models['catboost'] = CatBoostClassifier(
                iterations=300,
                learning_rate=0.1,
                depth=6,
                random_seed=random_state,
                verbose=0,
                thread_count=-1,
                loss_function='MultiClass'
            )
        elif use_catboost:
            print("⚠️  CatBoost не установлен. Пропускаем CatBoost в ансамбле.")

        # Мета-модель (level-1)
        if meta_model == 'logistic':
            self.meta_model = LogisticRegression(
                C=1.0,
                random_state=random_state,
                max_iter=1000,
                solver='lbfgs',
                multi_class='multinomial'
            )
        elif meta_model == 'svm':
            self.meta_model = SVC(
                C=1.0,
                random_state=random_state,
                probability=True,  # Для стекинга нужны вероятности
                kernel='linear'
            )
        elif meta_model == 'random_forest':
            self.meta_model = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                max_depth=None
            )
        else:
            raise ValueError("meta_model должен быть 'logistic', 'svm' или 'random_forest'")

        self.label_encoder = LabelEncoder()
        self.class_names = class_names
        self.use_blending = use_blending
        self.meta_model_type = meta_model
        self.text_field = text_field
        self.label_field = label_field
        self.is_trained = False
        self.num_classes = 0
        self.random_state = random_state
        self.use_catboost = use_catboost

        # Для хранения предсказаний при блендинге
        self.base_predictions = {}
        self.base_probabilities = {}

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

    def train_blending(self, train_data: List[Dict[str, Any]],
                       val_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Обучение с блендингом (используем отдельный validation set)
        """
        print("🎯 ОБУЧЕНИЕ С БЛЕНДИНГОМ...")

        X_train, y_train_raw = self.prepare_data(train_data)
        X_val, y_val_raw = self.prepare_data(val_data)

        # Кодируем метки
        if self.class_names is None:
            self.label_encoder.fit(y_train_raw + y_val_raw)
            self.class_names = list(self.label_encoder.classes_)
        else:
            self.label_encoder.fit(self.class_names)

        y_train = self.label_encoder.transform(y_train_raw)
        y_val = self.label_encoder.transform(y_val_raw)
        self.num_classes = len(self.class_names)

        # Векторизация
        print("📊 Векторизация текстов...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_val_vec = self.vectorizer.transform(X_val)

        print(f"   Размерность признаков: {X_train_vec.shape}")
        print(f"   Количество категорий: {self.num_classes}")
        print(f"   Базовые модели: {list(self.base_models.keys())}")
        print(f"   Мета-модель: {self.meta_model_type}")

        # Обучаем базовые модели на тренировочных данных
        base_val_probabilities = []

        print("\n🤖 ОБУЧЕНИЕ БАЗОВЫХ МОДЕЛЕЙ:")
        for name, model in self.base_models.items():
            print(f"   Обучение {name}...")

            try:
                if name == 'catboost' and CATBOOST_AVAILABLE:
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

                    # Получаем вероятности
                    if hasattr(model, 'predict_proba'):
                        val_prob = model.predict_proba(X_val_vec)
                    else:
                        # Для LinearSVC используем decision function
                        decision_scores = model.decision_function(X_val_vec)
                        val_prob = self._decision_to_probability_multiclass(decision_scores)

                accuracy = accuracy_score(y_val, val_pred)
                print(f"      ✅ Точность на val: {accuracy:.3f}")

                base_val_probabilities.append(val_prob)
                self.base_predictions[name] = val_pred
                self.base_probabilities[name] = val_prob

            except Exception as e:
                print(f"      ❌ Ошибка при обучении {name}: {e}")
                # Пропускаем эту модель
                if name in self.base_models:
                    del self.base_models[name]

        if not base_val_probabilities:
            raise ValueError("Не удалось обучить ни одну базовую модель")

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

        # Анализ производительности базовых моделей
        base_accuracies = {}
        for name, pred in self.base_predictions.items():
            base_accuracies[name] = accuracy_score(y_val, pred)

        return {
            'base_accuracies': base_accuracies,
            'meta_accuracy': meta_accuracy,
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }

    def train_stacking(self, train_data: List[Dict[str, Any]],
                       val_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Обучение со стекингом (используем кросс-валидацию)
        """
        print("🎯 ОБУЧЕНИЕ СО СТЕКИНГОМ...")

        X_train, y_train_raw = self.prepare_data(train_data)

        # Кодируем метки
        if self.class_names is None:
            self.label_encoder.fit(y_train_raw)
            self.class_names = list(self.label_encoder.classes_)
        else:
            self.label_encoder.fit(self.class_names)

        y_train = self.label_encoder.transform(y_train_raw)
        self.num_classes = len(self.class_names)

        # Векторизация
        print("📊 Векторизация текстов...")
        X_train_vec = self.vectorizer.fit_transform(X_train)

        print(f"   Размерность признаков: {X_train_vec.shape}")
        print(f"   Количество категорий: {self.num_classes}")
        print(f"   Базовые модели: {list(self.base_models.keys())}")
        print(f"   Мета-модель: {self.meta_model_type}")

        # Подготавливаем базовые модели для стекинга
        estimators = []
        for name, model in self.base_models.items():
            estimators.append((name, model))

        if not estimators:
            raise ValueError("Нет доступных базовых моделей для стекинга")

        # Создаем стекинг классификатор
        self.stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=self.meta_model,
            cv=3,
            passthrough=False,
            n_jobs=-1,
            verbose=0
        )

        print("\n🤖 ОБУЧЕНИЕ СТЕКИНГ МОДЕЛИ...")

        # Обучаем стекинг модель
        try:
            # Проверяем, есть ли CatBoost среди моделей
            has_catboost = any(name == 'catboost' for name in self.base_models.keys())

            if has_catboost and CATBOOST_AVAILABLE:
                X_train_dense = X_train_vec.toarray()
                self.stacking_model.fit(X_train_dense, y_train)
            else:
                self.stacking_model.fit(X_train_vec, y_train)
        except Exception as e:
            print(f"❌ Ошибка при обучении стекинга: {e}")
            # Пробуем без CatBoost
            if 'catboost' in self.base_models:
                print("⚠️  Пробуем без CatBoost...")
                del self.base_models['catboost']
                estimators = [(name, model) for name, model in self.base_models.items()]
                self.stacking_model = StackingClassifier(
                    estimators=estimators,
                    final_estimator=self.meta_model,
                    cv=3,
                    passthrough=False,
                    n_jobs=-1,
                    verbose=0
                )
                self.stacking_model.fit(X_train_vec, y_train)

        self.is_trained = True

        # Оценка на тренировочных данных
        if has_catboost and CATBOOST_AVAILABLE:
            train_pred = self.stacking_model.predict(X_train_dense)
        else:
            train_pred = self.stacking_model.predict(X_train_vec)

        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"✅ Точность на train: {train_accuracy:.3f}")

        # Оценка на валидации, если есть
        if val_data:
            val_accuracy, _ = self.evaluate(val_data, detailed=False)
            print(f"✅ Точность на val: {val_accuracy:.3f}")

        return {
            'train_accuracy': train_accuracy,
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }

    def train(self, train_data: List[Dict[str, Any]],
              val_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Основной метод обучения
        """
        if self.use_blending:
            if val_data is None:
                raise ValueError("Для блендинга необходим validation set")
            return self.train_blending(train_data, val_data)
        else:
            return self.train_stacking(train_data, val_data)

    def predict(self, texts: List[str]) -> Tuple[List[str], np.ndarray]:
        """
        Предсказание для списка текстов
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)

        if self.use_blending:
            predictions_encoded, probabilities = self._predict_blending(X_vec)
        else:
            predictions_encoded, probabilities = self._predict_stacking(X_vec)

        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        return predictions, probabilities

    def _predict_blending(self, X_vec: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказание для блендинга
        """
        # Получаем предсказания от всех базовых моделей
        base_probabilities = []

        for name, model in self.base_models.items():
            try:
                if name == 'catboost' and CATBOOST_AVAILABLE:
                    X_dense = X_vec.toarray()
                    prob = model.predict_proba(X_dense)
                else:
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_vec)
                    else:
                        decision_scores = model.decision_function(X_vec)
                        prob = self._decision_to_probability_multiclass(decision_scores)

                base_probabilities.append(prob)
            except Exception as e:
                print(f"⚠️  Ошибка в модели {name} при предсказании: {e}")
                continue

        if not base_probabilities:
            raise ValueError("Не удалось получить предсказания ни от одной модели")

        # Создаем мета-признаки
        meta_features = np.hstack(base_probabilities)

        # Предсказание мета-модели
        predictions_encoded = self.meta_model.predict(meta_features)

        # Получаем вероятности от мета-модели
        if hasattr(self.meta_model, 'predict_proba'):
            probabilities = self.meta_model.predict_proba(meta_features)
        else:
            # Для SVM без вероятностей
            decision_scores = self.meta_model.decision_function(meta_features)
            probabilities = self._decision_to_probability_multiclass(decision_scores)

        return predictions_encoded, probabilities

    def _predict_stacking(self, X_vec: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказание для стекинга
        """
        # Проверяем, есть ли CatBoost в моделях
        has_catboost = any(name == 'catboost' for name in self.base_models.keys())

        try:
            if has_catboost and CATBOOST_AVAILABLE:
                X_dense = X_vec.toarray()
                predictions_encoded = self.stacking_model.predict(X_dense)
            else:
                predictions_encoded = self.stacking_model.predict(X_vec)

            # Получаем вероятности
            if hasattr(self.stacking_model, 'predict_proba'):
                if has_catboost and CATBOOST_AVAILABLE:
                    probabilities = self.stacking_model.predict_proba(X_dense)
                else:
                    probabilities = self.stacking_model.predict_proba(X_vec)
            else:
                # Если predict_proba недоступен
                if has_catboost and CATBOOST_AVAILABLE:
                    decision_scores = self.stacking_model.decision_function(X_dense)
                else:
                    decision_scores = self.stacking_model.decision_function(X_vec)
                probabilities = self._decision_to_probability_multiclass(decision_scores)

            return predictions_encoded, probabilities

        except Exception as e:
            print(f"❌ Ошибка при предсказании стекинг моделью: {e}")
            raise

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

        # Получаем предсказания базовых моделей для анализа
        base_predictions = self._get_base_predictions(text)

        return {
            'prediction': pred,
            'prediction_encoded': pred_encoded,
            'category': pred,
            'category_probabilities': class_probs,
            'top_categories': top_categories,
            'confidence': prob[pred_encoded],
            'confidence_percent': prob[pred_encoded] * 100,
            'base_predictions': base_predictions,
            'consensus': self._get_consensus(base_predictions) if base_predictions else None
        }

    def _get_base_predictions(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Получает предсказания всех базовых моделей
        """
        X_vec = self.vectorizer.transform([text])
        base_results = {}

        if self.use_blending:
            # Для блендинга - используем отдельно обученные модели
            for name, model in self.base_models.items():
                try:
                    if name == 'catboost' and CATBOOST_AVAILABLE:
                        X_dense = X_vec.toarray()
                        pred_encoded = model.predict(X_dense)[0]
                        prob = model.predict_proba(X_dense)[0]
                    else:
                        pred_encoded = model.predict(X_vec)[0]
                        if hasattr(model, 'predict_proba'):
                            prob = model.predict_proba(X_vec)[0]
                        else:
                            # Для SVM без вероятностей
                            decision_score = model.decision_function(X_vec)
                            prob = self._decision_to_probability_multiclass(decision_score)[0]

                    pred = self.label_encoder.inverse_transform([pred_encoded])[0]

                    base_results[name] = {
                        'prediction': pred,
                        'prediction_encoded': pred_encoded,
                        'probability': prob,
                        'top_category': self.class_names[np.argmax(prob)],
                        'top_probability': np.max(prob)
                    }
                except Exception as e:
                    print(f"⚠️  Ошибка в модели {name}: {e}")
                    continue
        else:
            # Для стекинга - получаем из named_estimators_
            try:
                for name, model in self.stacking_model.named_estimators_.items():
                    try:
                        if name == 'catboost' and CATBOOST_AVAILABLE:
                            X_dense = X_vec.toarray()
                            pred_encoded = model.predict(X_dense)[0]
                            prob = model.predict_proba(X_dense)[0]
                        else:
                            pred_encoded = model.predict(X_vec)[0]
                            if hasattr(model, 'predict_proba'):
                                prob = model.predict_proba(X_vec)[0]
                            else:
                                decision_score = model.decision_function(X_vec)
                                prob = self._decision_to_probability_multiclass(decision_score)[0]

                        pred = self.label_encoder.inverse_transform([pred_encoded])[0]

                        base_results[name] = {
                            'prediction': pred,
                            'prediction_encoded': pred_encoded,
                            'probability': prob,
                            'top_category': self.class_names[np.argmax(prob)],
                            'top_probability': np.max(prob)
                        }
                    except Exception as e:
                        print(f"⚠️  Ошибка в модели {name}: {e}")
                        continue
            except Exception as e:
                print(f"⚠️  Ошибка при получении предсказаний базовых моделей: {e}")

        return base_results

    def _get_consensus(self, base_predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Анализ консенсуса базовых моделей
        """
        predictions = [data['prediction'] for data in base_predictions.values()]

        # Подсчитываем голоса для каждой категории
        vote_counts = {}
        for pred in predictions:
            vote_counts[pred] = vote_counts.get(pred, 0) + 1

        total_votes = len(predictions)
        max_votes = max(vote_counts.values()) if vote_counts else 0
        winning_category = max(vote_counts, key=vote_counts.get) if vote_counts else None

        return {
            'vote_counts': vote_counts,
            'total_votes': total_votes,
            'winning_category': winning_category,
            'max_votes': max_votes,
            'consensus_ratio': max_votes / total_votes if total_votes > 0 else 0,
            'unanimous': len(set(predictions)) == 1 if predictions else False
        }

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

        predictions, probabilities = self.predict(X_test)
        y_pred_encoded = self.label_encoder.transform(predictions)
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

    def analyze_model_performance(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Детальный анализ производительности всех моделей
        """
        X, y_raw = self.prepare_data(data)

        # Безопасное преобразование меток
        y = []
        for label in y_raw:
            if label in self.label_encoder.classes_:
                y.append(label)
            else:
                y.append(self.class_names[0])

        y_encoded = self.label_encoder.transform(y)
        X_vec = self.vectorizer.transform(X)

        print("\n📊 АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛЕЙ:")
        print("=" * 60)

        base_accuracies = {}

        if self.use_blending:
            # Для блендинга - базовые модели обучены отдельно
            for name, model in self.base_models.items():
                try:
                    if name == 'catboost' and CATBOOST_AVAILABLE:
                        X_dense = X_vec.toarray()
                        pred_encoded = model.predict(X_dense)
                    else:
                        pred_encoded = model.predict(X_vec)

                    accuracy = accuracy_score(y_encoded, pred_encoded)
                    base_accuracies[name] = accuracy
                    print(f"   {name.upper():<12}: {accuracy:.3f}")
                except Exception as e:
                    print(f"   {name.upper():<12}: ❌ Ошибка: {e}")
        else:
            # Для стекинга - получаем предсказания базовых моделей
            try:
                for name, model in self.stacking_model.named_estimators_.items():
                    try:
                        if name == 'catboost' and CATBOOST_AVAILABLE:
                            X_dense = X_vec.toarray()
                            pred_encoded = model.predict(X_dense)
                        else:
                            pred_encoded = model.predict(X_vec)

                        accuracy = accuracy_score(y_encoded, pred_encoded)
                        base_accuracies[name] = accuracy
                        print(f"   {name.upper():<12}: {accuracy:.3f}")
                    except Exception as e:
                        print(f"   {name.upper():<12}: ❌ Ошибка: {e}")
            except Exception as e:
                print(f"   ⚠️  Не удалось получить предсказания базовых моделей: {e}")

        # Оценка ансамбля
        ensemble_pred, _ = self.predict(X)
        ensemble_pred_encoded = self.label_encoder.transform(ensemble_pred)
        ensemble_accuracy = accuracy_score(y_encoded, ensemble_pred_encoded)
        print(f"   {'ENSEMBLE':<12}: {ensemble_accuracy:.3f}")

        # Улучшение по сравнению с лучшей базовой моделью
        improvement = 0
        if base_accuracies:
            best_base_accuracy = max(base_accuracies.values())
            improvement = ensemble_accuracy - best_base_accuracy
            print(f"\n   📈 Улучшение над лучшей моделью: {improvement:.3f}")
            if best_base_accuracy > 0:
                print(f"   📈 Относительное улучшение: {improvement / best_base_accuracy * 100:.1f}%")

        return {
            'base_accuracies': base_accuracies,
            'ensemble_accuracy': ensemble_accuracy,
            'improvement': improvement
        }

    def save_model(self, filename: str) -> None:
        """
        Сохранение модели
        """
        if self.use_blending:
            to_save = {
                'base_models': self.base_models,
                'meta_model': self.meta_model,
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder,
                'class_names': self.class_names,
                'use_blending': self.use_blending,
                'meta_model_type': self.meta_model_type,
                'text_field': self.text_field,
                'label_field': self.label_field,
                'num_classes': self.num_classes
            }
        else:
            to_save = {
                'stacking_model': self.stacking_model,
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder,
                'class_names': self.class_names,
                'use_blending': self.use_blending,
                'meta_model_type': self.meta_model_type,
                'text_field': self.text_field,
                'label_field': self.label_field,
                'num_classes': self.num_classes
            }

        joblib.dump(to_save, filename)
        print(f"💾 Модель сохранена: {filename}")

    def load_model(self, filename: str) -> None:
        """
        Загрузка модели
        """
        loaded = joblib.load(filename)

        self.vectorizer = loaded['vectorizer']
        self.label_encoder = loaded['label_encoder']
        self.class_names = loaded['class_names']
        self.num_classes = loaded['num_classes']
        self.use_blending = loaded['use_blending']
        self.meta_model_type = loaded['meta_model_type']
        self.text_field = loaded.get('text_field', 'text')
        self.label_field = loaded.get('label_field', 'category')

        if self.use_blending:
            self.base_models = loaded['base_models']
            self.meta_model = loaded['meta_model']
        else:
            self.stacking_model = loaded['stacking_model']

        self.is_trained = True
        print(f"📥 Модель загружена: {filename}")
        print(f"   Категории: {self.class_names}")
        print(f"   Количество категорий: {self.num_classes}")
        print(f"   Стратегия: {'Blending' if self.use_blending else 'Stacking'}")
        print(f"   Мета-модель: {self.meta_model_type}")


def compare_ensemble_strategies(train_data: List[Dict[str, Any]],
                                val_data: List[Dict[str, Any]],
                                test_data: List[Dict[str, Any]],
                                text_field: str = 'text',
                                label_field: str = 'category') -> Dict[str, Any]:
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
        ('stacking_svm', False, 'svm'),
        ('stacking_rf', False, 'random_forest'),
    ]

    results = {}

    for name, use_blending, meta_model in strategies:
        print(f"\n🎯 {name.upper()}:")
        try:
            ensemble = StackingCategoryClassifier(
                use_blending=use_blending,
                meta_model=meta_model,
                text_field=text_field,
                label_field=label_field,
                use_catboost=False  # Отключаем CatBoost для сравнения
            )

            if use_blending:
                ensemble.train(train_data, val_data)
            else:
                ensemble.train(train_data, val_data)

            test_accuracy, _ = ensemble.evaluate(test_data, detailed=False)

            # Анализ производительности
            performance = ensemble.analyze_model_performance(test_data)

            results[name] = {
                'model': ensemble,
                'accuracy': test_accuracy,
                'improvement': performance['improvement'] if 'improvement' in performance else 0
            }

            print(f"   ✅ Точность на тесте: {test_accuracy:.3f}")

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            results[name] = {
                'model': None,
                'accuracy': 0,
                'error': str(e)
            }

    # Сравнение результатов
    print("\n📊 ИТОГОВОЕ СРАВНЕНИЕ:")
    print("=" * 50)

    successful_results = {k: v for k, v in results.items() if 'error' not in v and v['model'] is not None}

    if successful_results:
        for name, result in sorted(successful_results.items(),
                                   key=lambda x: x[1]['accuracy'],
                                   reverse=True):
            improvement_info = ""
            if result['improvement'] > 0:
                improvement_info = f" (+{result['improvement']:.3f})"
            print(f"   {name:<25}: {result['accuracy']:.3f}{improvement_info}")
    else:
        print("   ❌ Ни одна стратегия не завершилась успешно")

    return results


def quick_train_ensemble(train_file: str,
                         val_file: str,
                         test_file: Optional[str] = None,
                         text_field: str = 'text',
                         label_field: str = 'category',
                         use_blending: bool = True,
                         meta_model: str = 'logistic',
                         use_catboost: bool = False,
                         output_model: str = 'ensemble_category_classifier.pkl') -> Optional[
    StackingCategoryClassifier]:
    """
    Быстрое обучение ансамбля из файлов
    """
    import json
    import os

    def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
        if not os.path.exists(filepath):
            print(f"⚠️  Файл не найден: {filepath}")
            return []
        with open(filepath, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    print("🚀 ЗАПУСК БЫСТРОГО ОБУЧЕНИЯ АНСАМБЛЯ")
    print("=" * 60)

    # Загрузка данных
    print(f"\n📥 Загрузка данных...")
    train_data = load_jsonl(train_file)
    if not train_data:
        print(f"❌ Ошибка: не удалось загрузить тренировочные данные из {train_file}")
        return None

    val_data = load_jsonl(val_file)
    if not val_data:
        print(f"❌ Ошибка: не удалось загрузить валидационные данные из {val_file}")
        return None

    print(f"   Train: {len(train_data)} примеров")
    print(f"   Val: {len(val_data)} примеров")

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
    print(f"\n🎯 Начало обучения ансамбля...")
    print(f"   Стратегия: {'Blending' if use_blending else 'Stacking'}")
    print(f"   Мета-модель: {meta_model}")
    print(f"   Использовать CatBoost: {'Да' if use_catboost else 'Нет'}")

    classifier = StackingCategoryClassifier(
        use_blending=use_blending,
        meta_model=meta_model,
        text_field=text_field,
        label_field=label_field,
        use_catboost=use_catboost
    )

    results = classifier.train(train_data, val_data)

    # Тестируем, если есть тестовые данные
    if test_data:
        print(f"\n🧪 Тестирование на тестовых данных...")
        accuracy, report = classifier.evaluate(test_data, detailed=True)
        print(f"\n🎯 Итоговая точность на тесте: {accuracy:.3f}")

        # Анализ производительности
        performance = classifier.analyze_model_performance(test_data)

        # Сохраняем отчет
        if report:
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv('ensemble_classification_report.csv', index=True)
            print(f"📄 Детальный отчет сохранен в 'ensemble_classification_report.csv'")

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

        if result['consensus']:
            print(f"   Консенсус моделей:")
            for category, votes in result['consensus']['vote_counts'].items():
                print(f"     {category}: {votes} голосов")

    return classifier


def main():
    """
    Пример использования стекинг/блендинг классификатора для многоклассовой классификации
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

        # 1. Тестирование блендинга
        print("\n" + "=" * 60)
        print("🎯 ТЕСТИРОВАНИЕ БЛЕНДИНГА")

        blending_classifier = StackingCategoryClassifier(
            use_blending=True,
            meta_model='logistic',
            text_field='text',
            label_field='category',
            use_catboost=False
        )

        blending_results = blending_classifier.train(train_data, val_data)

        # Анализ производительности
        print("\n📊 АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ БЛЕНДИНГА:")
        blending_classifier.analyze_model_performance(test_data)

        # 2. Тестирование стекинга
        print("\n" + "=" * 60)
        print("🎯 ТЕСТИРОВАНИЕ СТЕКИНГА")

        stacking_classifier = StackingCategoryClassifier(
            use_blending=False,
            meta_model='logistic',
            text_field='text',
            label_field='category',
            use_catboost=False
        )

        stacking_classifier.train(train_data, val_data)

        print("\n📊 АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ СТЕКИНГА:")
        stacking_classifier.analyze_model_performance(test_data)

        # 4. Сохранение моделей
        blending_classifier.save_model("blending_category_classifier.pkl")
        stacking_classifier.save_model("stacking_category_classifier.pkl")

        # 5. Сравнение всех стратегий (на подмножестве для скорости)
        print("\n" + "=" * 60)
        print("🔬 СРАВНЕНИЕ ВСЕХ СТРАТЕГИЙ (на подмножестве)")

        # Используем подмножество для скорости
        if len(train_data) > 300:
            train_subset = train_data[:300]
            val_subset = val_data[:100]
            test_subset = test_data[:100] if test_data else None
            print(f"   Используем подмножество: {len(train_subset)} train, "
                  f"{len(val_subset)} val")
        else:
            train_subset = train_data
            val_subset = val_data
            test_subset = test_data

        if test_subset:
            results = compare_ensemble_strategies(
                train_subset, val_subset, test_subset,
                'text', 'category'
            )

    except FileNotFoundError as e:
        print(f"❌ Файл не найден: {e}")
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Запуск примера
    print("🚀 ЗАПУСК ПРИМЕРА ИСПОЛЬЗОВАНИЯ АНСАМБЛЕЙ")
    print("=" * 60)
    main()