from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import joblib
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import pandas as pd

warnings.filterwarnings('ignore')

from util.jsonl_process import read_jsonl_basic


class VotingCategoryClassifier:
    """
    Классификатор категорий на основе голосования (Voting) для многоклассовой классификации
    """

    def __init__(self,
                 voting_type: str = 'soft',
                 class_names: Optional[List[str]] = None,
                 text_field: str = 'text',
                 label_field: str = 'category',
                 random_state: int = 42):
        """
        Args:
            voting_type: 'hard' или 'soft' голосование
            class_names: список названий классов (опционально)
            text_field: название поля с текстом
            label_field: название поля с меткой категории
            random_state: для воспроизводимости
        """
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            stop_words=None
        )

        # Создаем разнообразные модели для голосования
        self.models = {
            'logistic': LogisticRegression(
                C=1.0,
                random_state=random_state,
                max_iter=1000,
                multi_class='multinomial',
                solver='lbfgs'
            ),
            'svm_linear': SVC(
                C=1.0,
                kernel='linear',
                probability=True,  # Для soft voting
                random_state=random_state,
                decision_function_shape='ovr'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=random_state,
                n_jobs=-1
            ),
            'svm_rbf': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=random_state,
                decision_function_shape='ovr'
            ),
            'logistic_l2': LogisticRegression(
                C=0.1,
                penalty='l2',
                random_state=random_state,
                max_iter=1000,
                multi_class='multinomial',
                solver='lbfgs'
            )
        }

        self.voting_classifier = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting=voting_type,
            n_jobs=-1,
            verbose=0
        )

        self.label_encoder = LabelEncoder()
        self.class_names = class_names
        self.voting_type = voting_type
        self.text_field = text_field
        self.label_field = label_field
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
              val_data: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Обучение voting классификатора
        """
        print(f"🎯 ОБУЧЕНИЕ {self.voting_type.upper()} VOTING КЛАССИФИКАТОРА...")
        print(f"   Поле с текстом: '{self.text_field}'")
        print(f"   Поле с категорией: '{self.label_field}'")

        # Анализ распределения категорий
        train_dist = self.analyze_class_distribution(train_data)
        print(f"\n📊 РАСПРЕДЕЛЕНИЕ КАТЕГОРИЙ В TRAIN:")
        print(f"   Всего примеров: {train_dist['total_samples']}")
        print(f"   Количество категорий: {train_dist['num_classes']}")

        if train_dist['imbalance_ratio']:
            print(f"   Коэффициент дисбаланса: {train_dist['imbalance_ratio']:.2f}")

        # Подготовка данных
        X_train, y_train_raw = self.prepare_data(train_data)

        # Кодируем метки
        if self.class_names is None:
            self.label_encoder.fit(y_train_raw)
            self.class_names = list(self.label_encoder.classes_)
        else:
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
        print(f"   Тип голосования: {self.voting_type}")
        print(f"   Модели в ансамбле: {list(self.models.keys())}")
        print(f"   Количество моделей: {len(self.models)}")

        # Обучение voting классификатора
        print("\n🤖 Обучение ансамбля...")
        self.voting_classifier.fit(X_train_vec, y_train)
        self.is_trained = True

        # Оценка на тренировочных данных
        train_pred = self.voting_classifier.predict(X_train_vec)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"\n✅ Точность на train: {train_accuracy:.3f}")

        # Отчет по классам на train
        print("\n📊 ОТЧЕТ ПО КАТЕГОРИЯМ (train):")
        print(classification_report(y_train, train_pred, target_names=self.class_names))

        # Оценка индивидуальных моделей
        individual_accuracies = self._evaluate_individual_models(X_train_vec, y_train)

        # Оценка на валидации, если есть
        if val_data:
            val_accuracy, _ = self.evaluate(val_data, detailed=False)
            print(f"✅ Точность на val: {val_accuracy:.3f}")

        # Анализ ансамбля
        self._analyze_ensemble(X_train_vec, y_train, individual_accuracies)

        # Матрица ошибок на train
        if self.num_classes > 1 and self.num_classes <= 15:
            self._plot_confusion_matrix(y_train, train_pred, "Train Confusion Matrix")

    def _evaluate_individual_models(self, X_vec, y_true) -> Dict[str, float]:
        """
        Оценка индивидуальных моделей
        """
        print(f"\n📊 ПРОИЗВОДИТЕЛЬНОСТЬ ИНДИВИДУАЛЬНЫХ МОДЕЛЕЙ:")
        print("-" * 50)

        individual_accuracies = {}

        for name, model in self.models.items():
            try:
                # Обучаем модель отдельно если она еще не обучена
                # (в VotingClassifier модели уже обучены, но на всякий случай)
                if not hasattr(model, 'classes_'):
                    model.fit(X_vec, y_true)

                pred = model.predict(X_vec)
                accuracy = accuracy_score(y_true, pred)
                individual_accuracies[name] = accuracy
                print(f"   {name:<15}: {accuracy:.3f}")

            except Exception as e:
                print(f"   {name:<15}: ошибка - {e}")
                individual_accuracies[name] = 0

        return individual_accuracies

    def _analyze_ensemble(self, X_vec, y_true, individual_accuracies: Dict[str, float]):
        """
        Анализ работы ансамбля
        """
        print(f"\n📊 АНАЛИЗ {self.voting_type.upper()} VOTING АНСАМБЛЯ:")
        print("-" * 50)

        # Получаем предсказания всех моделей
        all_predictions = {}
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                try:
                    all_predictions[name] = model.predict(X_vec)
                except:
                    continue

        # Анализ согласованности
        n_samples = len(y_true)
        n_models = len(all_predictions)

        if n_models > 0:
            # Считаем единогласные решения
            unanimous_count = 0
            for i in range(n_samples):
                votes = [pred[i] for pred in all_predictions.values()]
                unique_votes = set(votes)
                if len(unique_votes) == 1:
                    unanimous_count += 1

            # Считаем согласие большинства
            majority_agree_count = 0
            for i in range(n_samples):
                votes = [pred[i] for pred in all_predictions.values()]
                from collections import Counter
                vote_counts = Counter(votes)
                majority_vote = max(vote_counts.values())
                if majority_vote >= n_models * 0.5:  # Более 50%
                    majority_agree_count += 1

            print(f"   Единогласные решения: {unanimous_count}/{n_samples} ({unanimous_count / n_samples * 100:.1f}%)")
            print(
                f"   Согласие большинства: {majority_agree_count}/{n_samples} ({majority_agree_count / n_samples * 100:.1f}%)")

        # Точность ансамбля
        ensemble_pred = self.voting_classifier.predict(X_vec)
        ensemble_accuracy = accuracy_score(y_true, ensemble_pred)
        print(f"   Точность ансамбля: {ensemble_accuracy:.3f}")

        # Сравнение с лучшей индивидуальной моделью
        if individual_accuracies:
            best_model_name = max(individual_accuracies, key=individual_accuracies.get)
            best_model_accuracy = individual_accuracies[best_model_name]
            improvement = ensemble_accuracy - best_model_accuracy

            print(f"   Лучшая индивидуальная модель: {best_model_name} ({best_model_accuracy:.3f})")
            print(f"   Улучшение ансамбля: {improvement:+.3f}")
            if best_model_accuracy > 0:
                print(f"   Относительное улучшение: {improvement / best_model_accuracy * 100:+.1f}%")

    def predict(self, texts: List[str]) -> Tuple[List[str], np.ndarray]:
        """
        Предсказание для списка текстов
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)
        predictions_encoded = self.voting_classifier.predict(X_vec)

        # Для soft voting получаем вероятности, для hard voting - вычисляем
        if self.voting_type == 'soft':
            probabilities = self.voting_classifier.predict_proba(X_vec)
        else:
            probabilities = self._get_hard_voting_probabilities(X_vec)

        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        return predictions, probabilities

    def _get_hard_voting_probabilities(self, X_vec) -> np.ndarray:
        """
        Вычисляет псевдо-вероятности для hard voting
        """
        # Получаем предсказания всех моделей
        all_predictions = []
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                try:
                    pred = model.predict(X_vec)
                    all_predictions.append(pred)
                except:
                    continue

        if not all_predictions:
            raise Exception("Не удалось получить предсказания моделей")

        all_predictions = np.array(all_predictions)
        n_models = len(all_predictions)
        n_samples = len(all_predictions[0])

        # Вычисляем вероятности на основе голосования для каждого класса
        probabilities = np.zeros((n_samples, self.num_classes))

        for i in range(n_samples):
            votes = all_predictions[:, i]
            for class_idx in range(self.num_classes):
                class_votes = np.sum(votes == class_idx)
                probabilities[i, class_idx] = class_votes / n_models

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

        # Получаем детальную информацию о голосовании
        voting_details = self._get_voting_details(text)

        return {
            'prediction': pred,
            'category': pred,
            'prediction_encoded': pred_encoded,
            'category_probabilities': class_probs,
            'top_categories': top_categories,
            'confidence': prob[pred_encoded],
            'confidence_percent': prob[pred_encoded] * 100,
            'voting_type': self.voting_type,
            'voting_details': voting_details
        }

    def _get_voting_details(self, text: str) -> Dict[str, Any]:
        """
        Получает детальную информацию о голосовании моделей
        """
        X_vec = self.vectorizer.transform([text])

        voting_results = {}
        all_predictions = []
        all_probabilities = []

        for name, model in self.models.items():
            try:
                pred_encoded = model.predict(X_vec)[0]
                pred = self.label_encoder.inverse_transform([pred_encoded])[0]

                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_vec)[0]
                else:
                    # Для моделей без вероятностей создаем равномерное распределение
                    prob = np.ones(self.num_classes) / self.num_classes

                voting_results[name] = {
                    'prediction': pred,
                    'prediction_encoded': pred_encoded,
                    'probability': prob,
                    'top_category': self.class_names[np.argmax(prob)],
                    'top_probability': np.max(prob),
                    'confidence': np.max(prob)
                }

                all_predictions.append(pred_encoded)
                all_probabilities.append(prob)

            except Exception as e:
                print(f"⚠️  Ошибка в модели {name}: {e}")
                continue

        # Анализ голосования
        if all_predictions:
            # Подсчет голосов для каждой категории
            vote_counts = {}
            for pred_encoded in all_predictions:
                pred = self.label_encoder.inverse_transform([pred_encoded])[0]
                vote_counts[pred] = vote_counts.get(pred, 0) + 1

            total_votes = len(all_predictions)
            max_votes = max(vote_counts.values()) if vote_counts else 0
            winning_category = max(vote_counts, key=vote_counts.get) if vote_counts else None

            if self.voting_type == 'soft':
                # Для soft voting вычисляем средние вероятности
                avg_probabilities = np.mean(all_probabilities, axis=0)
                winning_idx = np.argmax(avg_probabilities)
                decision_reason = f"Soft voting (средняя вероятность: {avg_probabilities[winning_idx]:.3f})"
            else:
                decision_reason = f"Hard voting ({max_votes}/{total_votes} за {winning_category})"

            return {
                'individual_votes': voting_results,
                'vote_counts': vote_counts,
                'total_votes': total_votes,
                'winning_category': winning_category,
                'max_votes': max_votes,
                'consensus_ratio': max_votes / total_votes if total_votes > 0 else 0,
                'unanimous': len(set(all_predictions)) == 1 if all_predictions else False,
                'decision_reason': decision_reason
            }
        else:
            return {
                'individual_votes': {},
                'vote_counts': {},
                'total_votes': 0,
                'winning_category': None,
                'max_votes': 0,
                'consensus_ratio': 0,
                'unanimous': False,
                'decision_reason': 'Нет доступных моделей'
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
                y_test.append(self.class_names[0])

        y_test_encoded = self.label_encoder.transform(y_test)
        X_test_vec = self.vectorizer.transform(X_test)

        y_pred_encoded = self.voting_classifier.predict(X_test_vec)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)

        if detailed:
            print(f"\n📊 ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ:")
            print(f"   Примеров: {len(test_data)}")
            print(f"   Тип голосования: {self.voting_type}")
            print(f"   Точность: {accuracy:.3f}")

            print(f"\n📈 ДЕТАЛЬНЫЙ ОТЧЕТ ПО КАТЕГОРИЯМ:")
            print(classification_report(y_test_encoded, y_pred_encoded,
                                        target_names=self.class_names, digits=3))

            # Матрица ошибок
            if self.num_classes > 1 and self.num_classes <= 15:
                print(f"\n📊 МАТРИЦА ОШИБОК:")
                cm = confusion_matrix(y_test_encoded, y_pred_encoded)
                self._print_confusion_matrix(cm)

                if plot_confusion_matrix:
                    self._plot_confusion_matrix(y_test_encoded, y_pred_encoded,
                                                f"Test Confusion Matrix ({self.voting_type} Voting)")

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
            if self.num_classes <= 1 or self.num_classes > 15:
                return

            cm = confusion_matrix(y_true, y_pred)

            plt.figure(figsize=(max(10, min(self.num_classes, 12)),
                                max(8, min(self.num_classes * 0.8, 10))))

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

            filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(' ', '_')
            plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"⚠️  Не удалось построить матрицу ошибок: {e}")

    def compare_with_individual_models(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Сравнение ансамбля с индивидуальными моделями
        """
        X_test, y_test_raw = self.prepare_data(test_data)

        # Безопасное преобразование меток
        y_test = []
        for label in y_test_raw:
            if label in self.label_encoder.classes_:
                y_test.append(label)
            else:
                y_test.append(self.class_names[0])

        y_test_encoded = self.label_encoder.transform(y_test)
        X_test_vec = self.vectorizer.transform(X_test)

        print(f"\n🔬 СРАВНЕНИЕ {self.voting_type.upper()} VOTING С ИНДИВИДУАЛЬНЫМИ МОДЕЛЯМИ:")
        print("=" * 60)

        # Точность ансамбля
        ensemble_pred_encoded = self.voting_classifier.predict(X_test_vec)
        ensemble_accuracy = accuracy_score(y_test_encoded, ensemble_pred_encoded)
        print(f"   {'VOTING ENSEMBLE':<20}: {ensemble_accuracy:.3f}")

        # Точности индивидуальных моделей
        individual_accuracies = {}
        for name, model in self.models.items():
            try:
                pred_encoded = model.predict(X_test_vec)
                accuracy = accuracy_score(y_test_encoded, pred_encoded)
                individual_accuracies[name] = accuracy
                print(f"   {name:<20}: {accuracy:.3f}")
            except Exception as e:
                print(f"   {name:<20}: ошибка - {e}")

        # Анализ улучшения
        comparison_result = {}
        if individual_accuracies:
            best_individual_name = max(individual_accuracies, key=individual_accuracies.get)
            best_individual_accuracy = individual_accuracies[best_individual_name]
            improvement = ensemble_accuracy - best_individual_accuracy

            print(f"\n   📈 Улучшение над лучшей моделью ({best_individual_name}): {improvement:.3f}")
            if best_individual_accuracy > 0:
                print(f"   📈 Относительное улучшение: {improvement / best_individual_accuracy * 100:.1f}%")

            comparison_result = {
                'ensemble_accuracy': ensemble_accuracy,
                'individual_accuracies': individual_accuracies,
                'improvement': improvement,
                'best_individual_name': best_individual_name,
                'best_individual_accuracy': best_individual_accuracy
            }

        return comparison_result

    def save_model(self, filename: str) -> None:
        """
        Сохранение модели
        """
        joblib.dump({
            'voting_classifier': self.voting_classifier,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'class_names': self.class_names,
            'models': self.models,
            'voting_type': self.voting_type,
            'text_field': self.text_field,
            'label_field': self.label_field,
            'num_classes': self.num_classes,
            'random_state': self.random_state
        }, filename)
        print(f"💾 Voting модель сохранена: {filename}")

    def load_model(self, filename: str) -> None:
        """
        Загрузка модели
        """
        loaded = joblib.load(filename)
        self.voting_classifier = loaded['voting_classifier']
        self.vectorizer = loaded['vectorizer']
        self.label_encoder = loaded['label_encoder']
        self.class_names = loaded['class_names']
        self.models = loaded['models']
        self.voting_type = loaded['voting_type']
        self.text_field = loaded.get('text_field', 'text')
        self.label_field = loaded.get('label_field', 'category')
        self.num_classes = loaded.get('num_classes', len(self.class_names))
        self.random_state = loaded.get('random_state', 42)
        self.is_trained = True

        print(f"📥 Voting модель загружена: {filename}")
        print(f"   Категории: {self.class_names}")
        print(f"   Количество категорий: {self.num_classes}")
        print(f"   Тип голосования: {self.voting_type}")


def compare_voting_strategies(train_data: List[Dict[str, Any]],
                              val_data: List[Dict[str, Any]],
                              test_data: List[Dict[str, Any]],
                              text_field: str = 'text',
                              label_field: str = 'category') -> Dict[str, Any]:
    """
    Сравнение Hard Voting и Soft Voting
    """
    print("🔬 СРАВНЕНИЕ HARD VS SOFT VOTING")
    print("=" * 60)

    results = {}

    for voting_type in ['hard', 'soft']:
        print(f"\n🎯 {voting_type.upper()} VOTING:")
        voting_classifier = VotingCategoryClassifier(
            voting_type=voting_type,
            text_field=text_field,
            label_field=label_field
        )

        # Используем подмножество если данных много
        if len(train_data) > 500:
            train_subset = train_data[:500]
            print(f"   Используем подмножество из {len(train_subset)} примеров")
        else:
            train_subset = train_data

        voting_classifier.train(train_subset, val_data)

        # Сравнение с индивидуальными моделями
        comparison = voting_classifier.compare_with_individual_models(test_data)

        # Оценка на тесте
        test_accuracy, _ = voting_classifier.evaluate(test_data, detailed=False)

        results[voting_type] = {
            'classifier': voting_classifier,
            'comparison': comparison,
            'test_accuracy': test_accuracy
        }

    # Итоговое сравнение
    print("\n📊 ИТОГОВОЕ СРАВНЕНИЕ:")
    print("=" * 40)

    for voting_type, result in results.items():
        accuracy = result['test_accuracy']
        improvement = result['comparison'].get('improvement', 0) if result['comparison'] else 0
        print(f"   {voting_type.upper():<12} Voting: {accuracy:.3f} (улучшение: {improvement:+.3f})")

    return results


def quick_train_voting(train_file: str,
                       val_file: Optional[str] = None,
                       test_file: Optional[str] = None,
                       text_field: str = 'text',
                       label_field: str = 'category',
                       voting_type: str = 'soft',
                       output_model: str = 'voting_category_classifier.pkl') -> Optional[VotingCategoryClassifier]:
    """
    Быстрое обучение Voting модели из файлов
    """
    import json
    import os

    def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
        if not os.path.exists(filepath):
            print(f"⚠️  Файл не найден: {filepath}")
            return []
        with open(filepath, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    print("🚀 ЗАПУСК БЫСТРОГО ОБУЧЕНИЯ VOTING КЛАССИФИКАТОРА")
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
    print(f"\n🎯 Начало обучения Voting классификатора...")
    print(f"   Тип голосования: {voting_type}")

    classifier = VotingCategoryClassifier(
        voting_type=voting_type,
        text_field=text_field,
        label_field=label_field
    )

    classifier.train(train_data, val_data)

    # Тестируем, если есть тестовые данные
    if test_data:
        print(f"\n🧪 Тестирование на тестовых данных...")
        accuracy, report = classifier.evaluate(test_data, detailed=True)
        print(f"\n🎯 Итоговая точность на тесте: {accuracy:.3f}")

        # Сравнение с индивидуальными моделями
        comparison = classifier.compare_with_individual_models(test_data)

        # Сохраняем отчет
        if report:
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv('voting_classification_report.csv', index=True)
            print(f"📄 Детальный отчет сохранен в 'voting_classification_report.csv'")

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
        print(f"   Тип голосования: {result['voting_type']}")

        if result['top_categories']:
            print(f"   Топ-3 категории:")
            for i, cat in enumerate(result['top_categories'], 1):
                print(f"     {i}. {cat['category']}: {cat['probability_percent']:.1f}%")

        if result['voting_details']:
            print(f"   Детали голосования:")
            for model_name, vote in result['voting_details']['individual_votes'].items():
                print(f"     {model_name}: {vote['prediction']} (уверенность: {vote['confidence']:.3f})")

    return classifier


def main():
    """
    Пример использования Voting классификатора для многоклассовой классификации
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

        # 1. Тестирование Soft Voting
        print("\n" + "=" * 60)
        print("🎯 ТЕСТИРОВАНИЕ SOFT VOTING")

        soft_voting = VotingCategoryClassifier(voting_type='soft')

        # Используем подмножество если данных много
        if len(train_data) > 500:
            train_subset = train_data[:500]
            print(f"ℹ️  Используем подмножество из {len(train_subset)} примеров")
        else:
            train_subset = train_data

        soft_voting.train(train_subset, val_data)

        # 2. Тестирование Hard Voting
        print("\n" + "=" * 60)
        print("🎯 ТЕСТИРОВАНИЕ HARD VOTING")

        hard_voting = VotingCategoryClassifier(voting_type='hard')
        hard_voting.train(train_subset, val_data)

        # 3. Сравнение стратегий
        print("\n" + "=" * 60)
        results = compare_voting_strategies(train_data, val_data, test_data, 'text', 'category')

        # 5. Сохранение моделей
        soft_voting.save_model("soft_voting_category_classifier.pkl")
        hard_voting.save_model("hard_voting_category_classifier.pkl")

    except FileNotFoundError as e:
        print(f"❌ Файл не найден: {e}")
        print("ℹ️  Проверьте пути к файлам данных")
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Запуск примера
    print("🚀 ЗАПУСК ПРИМЕРА ИСПОЛЬЗОВАНИЯ VOTING КЛАССИФИКАТОРА")
    print("=" * 60)
    main()