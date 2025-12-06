from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    hamming_loss,
    f1_score,
    precision_score,
    recall_score,
    multilabel_confusion_matrix
)
import numpy as np
import joblib
import pandas as pd
from scipy.stats import randint, uniform
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings('ignore')


class MultiLabelTextClassifier:
    """
    Классификатор для многометочной (multilabel) классификации текстов
    """

    def __init__(self, max_training_time=300, n_iter=50, random_state=42):
        """
        Args:
            max_training_time: максимальное время обучения (в секундах)
            n_iter: количество итераций случайного поиска
            random_state: для воспроизводимости
        """
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words=None,
            analyzer='word'
        )

        # Инициализируем MultiLabelBinarizer для преобразования меток
        self.label_binarizer = MultiLabelBinarizer()

        # Определяем модели и пространства параметров для поиска
        # Для multilabel используем OneVsRestClassifier
        self.models = {
            'logistic': {
                'model': OneVsRestClassifier(LogisticRegression(random_state=random_state, max_iter=1000)),
                'params': {
                    'estimator__C': uniform(0.001, 100),
                    'estimator__penalty': ['l1', 'l2'],
                    'estimator__solver': ['liblinear', 'saga'],
                }
            },
            'svm_linear': {
                'model': OneVsRestClassifier(LinearSVC(random_state=random_state, dual=False)),
                'params': {
                    'estimator__C': uniform(0.1, 10),
                    'estimator__loss': ['squared_hinge'],
                }
            },
            'random_forest': {
                'model': OneVsRestClassifier(RandomForestClassifier(random_state=random_state)),
                'params': {
                    'estimator__n_estimators': randint(50, 300),
                    'estimator__max_depth': [None, 10, 20, 30],
                    'estimator__min_samples_split': randint(2, 20),
                }
            },
            'naive_bayes': {
                'model': OneVsRestClassifier(MultinomialNB()),
                'params': {
                    'estimator__alpha': uniform(0.001, 2.0)
                }
            },
        }

        self.max_training_time = max_training_time
        self.n_iter = n_iter
        self.random_state = random_state
        self.is_trained = False
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.label_names = None
        self.history = {
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1': [],
            'val_f1': [],
            'train_hamming': [],
            'val_hamming': []
        }

        print(f"🚀 Multi-label Classifier инициализирован:")
        print(f"   Максимальное время: {max_training_time} сек")
        print(f"   Итераций поиска: {n_iter}")
        print(f"   Модели для тестирования: {list(self.models.keys())}")

    def prepare_data(self, data):
        """
        Подготовка данных: извлекаем тексты и метки
        """
        texts = [item['text'] for item in data]

        # Предполагаем, что метки хранятся в поле 'binary_labels'
        labels = [item['binary_labels'] for item in data]

        # Преобразуем в бинарный формат
        if not hasattr(self.label_binarizer, 'classes_'):
            labels_binary = self.label_binarizer.fit_transform(labels)
            # Сохраняем названия классов как строки
            self.label_names = [str(cls) for cls in self.label_binarizer.classes_]
            print(f"   Количество классов: {len(self.label_names)}")
            print(f"   Названия классов: {self.label_names}")
        else:
            labels_binary = self.label_binarizer.transform(labels)

        return texts, labels_binary

    def train(self, train_data, val_data=None):
        """
        Обучение классификатора
        """
        print("🎯 АВТОМАТИЗИРОВАННЫЙ ПОДБОР МОДЕЛЕЙ...")

        # Подготовка данных
        X_train, y_train = self.prepare_data(train_data)

        # Векторизация текстов
        print("📊 Векторизация текстов...")
        X_train_vec = self.vectorizer.fit_transform(X_train)

        print(f"   Размерность признаков: {X_train_vec.shape}")
        print(f"   Размерность меток: {y_train.shape}")
        print(f"   Количество примеров: {len(y_train)}")

        # Подготовка валидационных данных если есть
        if val_data:
            X_val, y_val = self.prepare_data(val_data)
            X_val_vec = self.vectorizer.transform(X_val)
        else:
            X_val_vec, y_val = None, None

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
                    scoring='f1_weighted',  # Используем weighted F1 для multilabel
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=0
                )

                search.fit(X_train_vec, y_train)

                # Оценка на валидации если есть
                val_score = None
                if val_data:
                    y_val_pred = search.best_estimator_.predict(X_val_vec)
                    val_score = f1_score(y_val, y_val_pred, average='weighted')

                best_models[model_name] = {
                    'model': search.best_estimator_,
                    'train_score': search.best_score_,
                    'val_score': val_score,
                    'params': search.best_params_
                }

                print(f"      ✅ Train F1: {search.best_score_:.3f}" +
                      (f", Val F1: {val_score:.3f}" if val_score else ""))

            except Exception as e:
                print(f"      ❌ Ошибка при поиске для {model_name}: {e}")
                continue

        # Выбираем лучшую модель
        if best_models:
            # Используем валидационный score если есть, иначе train score
            score_key = 'val_score' if val_data else 'train_score'

            # Исправленная строка: используем score_key вместо score_score
            self.best_model_name = max(
                best_models.keys(),
                key=lambda x: best_models[x][score_key] if best_models[x][score_key] is not None else 0
            )

            self.best_model = best_models[self.best_model_name]['model']
            self.best_score = best_models[self.best_model_name][score_key]
            self.best_params = best_models[self.best_model_name]['params']

            print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {self.best_model_name}")
            print(f"   F1 Score: {self.best_score:.3f}")
            print(f"   Параметры: {self.best_params}")

            self.is_trained = True

            # Показываем сравнение всех моделей
            self._show_model_comparison(best_models)

            # Сохраняем историю для графиков
            if val_data:
                y_train_pred = self.best_model.predict(X_train_vec)
                y_val_pred = self.best_model.predict(X_val_vec)

                self.history['train_f1'].append(f1_score(y_train, y_train_pred, average='weighted'))
                self.history['val_f1'].append(f1_score(y_val, y_val_pred, average='weighted'))
                self.history['train_accuracy'].append(accuracy_score(y_train, y_train_pred))
                self.history['val_accuracy'].append(accuracy_score(y_val, y_val_pred))

                # Вычисляем Hamming loss
                self.history['train_hamming'] = [hamming_loss(y_train, y_train_pred)]
                self.history['val_hamming'] = [hamming_loss(y_val, y_val_pred)]

                print(f"✅ Точность на train: {self.history['train_accuracy'][-1]:.3f}")
                print(f"✅ Точность на val: {self.history['val_accuracy'][-1]:.3f}")
                print(f"✅ Hamming loss на train: {self.history['train_hamming'][-1]:.3f}")
                print(f"✅ Hamming loss на val: {self.history['val_hamming'][-1]:.3f}")
        else:
            raise Exception("Не удалось обучить ни одну модель!")

    def _show_model_comparison(self, best_models):
        """
        Показывает сравнение всех протестированных моделей
        """
        print(f"\n📊 СРАВНЕНИЕ МОДЕЛЕЙ:")
        print("-" * 50)

        # Сортируем модели по валидационному score если есть, иначе по train score
        sorted_models = sorted(best_models.items(),
                               key=lambda x: x[1]['val_score'] if x[1]['val_score'] is not None else x[1][
                                   'train_score'],
                               reverse=True)

        for model_name, results in sorted_models:
            train_score = results['train_score']
            val_score = results['val_score']

            score_str = f"Train F1: {train_score:.3f}"
            if val_score is not None:
                score_str += f", Val F1: {val_score:.3f}"

            print(f"   {model_name:<15}: {score_str}")

    def predict(self, texts, threshold=0.5):
        """
        Предсказание для списка текстов
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)

        # Для multilabel получаем вероятности и применяем порог
        if hasattr(self.best_model, "predict_proba"):
            probabilities = self.best_model.predict_proba(X_vec)
            predictions = (probabilities >= threshold).astype(int)
        else:
            # Для моделей без predict_proba (например, LinearSVC)
            predictions = self.best_model.predict(X_vec)
            probabilities = None

        # Преобразуем обратно в список меток
        predictions_labels = self.label_binarizer.inverse_transform(predictions)

        return predictions, predictions_labels, probabilities

    def predict_single(self, text, threshold=0.5):
        """
        Предсказание для одного текста
        """
        predictions, pred_labels, probs = self.predict([text], threshold)

        result = {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'predicted_labels': list(pred_labels[0]) if len(pred_labels[0]) > 0 else [],
            'model_type': type(self.best_model).__name__,
            'model_name': self.best_model_name
        }

        if probs is not None:
            result['probabilities'] = {
                self.label_names[i]: float(probs[0][i])
                for i in range(len(self.label_names))
            }

        return result

    def evaluate(self, test_data, threshold=0.5):
        """
        Оценка модели на тестовых данных
        """
        X_test, y_test = self.prepare_data(test_data)
        X_test_vec = self.vectorizer.transform(X_test)

        # Используем прямой predict для получения бинарных предсказаний
        if hasattr(self.best_model, "predict_proba"):
            probabilities = self.best_model.predict_proba(X_test_vec)
            y_pred = (probabilities >= threshold).astype(int)
        else:
            y_pred = self.best_model.predict(X_test_vec)

        # Вычисляем метрики
        accuracy = accuracy_score(y_test, y_pred)
        h_loss = hamming_loss(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_samples = f1_score(y_test, y_pred, average='samples')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        print("\n📊 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        print("-" * 50)
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Hamming Loss: {h_loss:.3f}")
        print(f"Precision (weighted): {precision:.3f}")
        print(f"Recall (weighted): {recall:.3f}")
        print(f"F1 Macro: {f1_macro:.3f}")
        print(f"F1 Weighted: {f1_weighted:.3f}")
        print(f"F1 Samples: {f1_samples:.3f}")

        # Подробный отчет по классам
        print("\n📈 CLASSIFICATION REPORT (по каждому классу):")

        # Для multilabel классификации classification_report работает по-другому
        # Нужно выводить отчет для каждого класса отдельно
        for i, label_name in enumerate(self.label_names):
            print(f"\n--- Класс: {label_name} ---")
            try:
                # Для каждого класса выводим метрики бинарной классификации
                y_test_class = y_test[:, i]
                y_pred_class = y_pred[:, i]

                # Вычисляем метрики для текущего класса
                if len(np.unique(y_test_class)) > 1:  # Только если есть оба класса
                    print(f"Precision: {precision_score(y_test_class, y_pred_class, zero_division=0):.3f}")
                    print(f"Recall: {recall_score(y_test_class, y_pred_class, zero_division=0):.3f}")
                    print(f"F1: {f1_score(y_test_class, y_pred_class, zero_division=0):.3f}")
                else:
                    print("Только один класс присутствует в тестовых данных")

                # Матрица ошибок для класса
                tn, fp, fn, tp = confusion_matrix(y_test_class, y_pred_class).ravel()
                print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

            except Exception as e:
                print(f"Ошибка при вычислении метрик для класса {label_name}: {e}")

        # Матрицы ошибок для каждого класса
        if self.label_names is not None and len(self.label_names) <= 15:
            print("\n🔍 МАТРИЦЫ ОШИБОК ПО КЛАССАМ (первые 5 классов):")
            cm = multilabel_confusion_matrix(y_test, y_pred)

            for i, (class_name, class_cm) in enumerate(zip(self.label_names, cm)):
                if i < 5:  # Показываем только первые 5 классов чтобы не перегружать вывод
                    print(f"\nКласс {i + 1}: {class_name}")
                    print(f"              Предсказан 0  Предсказан 1")
                    print(f"Реально 0:     {class_cm[0][0]:^10}    {class_cm[0][1]:^10}")
                    print(f"Реально 1:     {class_cm[1][0]:^10}    {class_cm[1][1]:^10}")
                    tn, fp, fn, tp = class_cm.ravel()
                    if (tp + fp) > 0:
                        print(f"Precision: {tp / (tp + fp):.3f}")
                    if (tp + fn) > 0:
                        print(f"Recall: {tp / (tp + fn):.3f}")
            if len(self.label_names) > 5:
                print(f"\n... и еще {len(self.label_names) - 5} классов")

        return {
            'accuracy': accuracy,
            'hamming_loss': h_loss,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_samples': f1_samples,
            'precision': precision,
            'recall': recall
        }

    def plot_training_history(self, save_path=None):
        """
        Построение графиков обучения
        """
        if not self.history['train_accuracy']:
            print("⚠️ Нет данных истории обучения")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # График точности
        axes[0].plot(self.history['train_accuracy'], label='Train Accuracy', marker='o', linewidth=2)
        if self.history['val_accuracy']:
            axes[0].plot(self.history['val_accuracy'], label='Val Accuracy', marker='s', linewidth=2)
        axes[0].set_title('Accuracy')
        axes[0].set_xlabel('Model Selection')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # График F1-score
        axes[1].plot(self.history['train_f1'], label='Train F1', marker='o', linewidth=2)
        if self.history['val_f1']:
            axes[1].plot(self.history['val_f1'], label='Val F1', marker='s', linewidth=2)
        axes[1].set_title('F1 Score (weighted)')
        axes[1].set_xlabel('Model Selection')
        axes[1].set_ylabel('F1 Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # График Hamming Loss если есть
        if self.history.get('train_hamming'):
            axes[2].plot(self.history['train_hamming'], label='Train Hamming Loss', marker='o', linewidth=2)
            if self.history.get('val_hamming'):
                axes[2].plot(self.history['val_hamming'], label='Val Hamming Loss', marker='s', linewidth=2)
            axes[2].set_title('Hamming Loss')
            axes[2].set_xlabel('Model Selection')
            axes[2].set_ylabel('Hamming Loss')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'Best Model: {self.best_model_name}', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Графики сохранены: {save_path}")

        plt.show()

    def plot_confusion_matrices(self, test_data, max_classes=12, save_path=None):
        """
        Визуализация матриц ошибок для каждого класса
        """
        if self.label_names is None:
            print("⚠️ Нет информации о классах")
            return

        X_test, y_test = self.prepare_data(test_data)
        X_test_vec = self.vectorizer.transform(X_test)

        y_pred = self.best_model.predict(X_test_vec)

        cm = multilabel_confusion_matrix(y_test, y_pred)

        n_classes = min(len(self.label_names), max_classes)
        if n_classes == 0:
            print("⚠️ Нет классов для отображения")
            return

        n_cols = min(4, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))

        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, (class_name, class_cm) in enumerate(zip(self.label_names[:n_classes], cm[:n_classes])):
            ax = axes[i]
            sns.heatmap(class_cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        cbar_kws={'shrink': 0.8})
            ax.set_title(f'Class: {str(class_name)[:20]}', fontsize=10)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(['0', '1'])
            ax.set_yticklabels(['0', '1'])

        # Скрываем лишние subplots
        for i in range(n_classes, len(axes)):
            axes[i].axis('off')

        plt.suptitle('Confusion Matrices for Each Class', fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Матрицы ошибок сохранены: {save_path}")

        plt.show()

    def get_model_info(self):
        """
        Возвращает информацию о модели
        """
        if not self.is_trained:
            return {"error": "Модель не обучена"}

        # Исправление: проверяем что label_names не None перед использованием len()
        label_names_list = self.label_names if self.label_names is not None else []

        return {
            'model_name': self.best_model_name,
            'model_type': type(self.best_model).__name__,
            'best_score': self.best_score,
            'parameters': self.best_params,
            'feature_count': len(self.vectorizer.get_feature_names_out()),
            'label_count': len(label_names_list),
            'labels': label_names_list
        }

    def save_model(self, filename):
        """
        Сохранение модели
        """
        # Преобразуем label_names в список перед сохранением
        label_names_to_save = list(self.label_names) if self.label_names is not None else None

        joblib.dump({
            'best_model': self.best_model,
            'vectorizer': self.vectorizer,
            'label_binarizer': self.label_binarizer,
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'label_names': label_names_to_save,  # Сохраняем как список
            'history': self.history
        }, filename, compress=3)
        print(f"💾 Multi-label модель сохранена: {filename}")

    def load_model(self, filename):
        """
        Загрузка модели
        """
        loaded = joblib.load(filename)
        self.best_model = loaded['best_model']
        self.vectorizer = loaded['vectorizer']
        self.label_binarizer = loaded['label_binarizer']
        self.best_model_name = loaded['best_model_name']
        self.best_score = loaded.get('best_score', 0)
        self.best_params = loaded.get('best_params', {})

        # Загружаем label_names и преобразуем в список если нужно
        loaded_label_names = loaded.get('label_names', None)
        if loaded_label_names is not None:
            self.label_names = [str(cls) for cls in loaded_label_names]  # Гарантируем строки
        else:
            self.label_names = None

        self.history = loaded.get('history', {})
        self.is_trained = True
        print(f"📥 Multi-label модель загружена: {filename}")


# Пример использования
def main():
    """
    Пример использования многометочного классификатора
    """
    # Загрузка данных (предполагается, что у вас уже есть функция read_jsonl_basic)
    try:
        from util.jsonl_process import read_jsonl_basic

        train_data = read_jsonl_basic('../util/news_multilabel_train_data.jsonl')
        val_data = read_jsonl_basic('../util/news_multilabel_val_data.jsonl')
        test_data = read_jsonl_basic('../util/news_multilabel_test_data.jsonl')

        print(f"📊 Данные: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

        # Пример первой записи для понимания структуры
        if train_data:
            print(f"\n📝 Пример записи:")
            print(f"   Text length: {len(train_data[0]['text'])} chars")
            print(f"   Labels: {train_data[0]['binary_labels']}")
            print(f"   Number of labels: {len(train_data[0]['binary_labels'])}")
    except Exception as e:
        print(f"⚠️ Не удалось загрузить данные: {e}")
        print("Создаем тестовые данные для демонстрации...")
        # Создаем тестовые данные для демонстрации
        train_data = [
            {"text": "Пример текста 1 о спорте и политике", "binary_labels": [1, 0, 1, 0, 1, 0]},
            {"text": "Пример текста 2 об экономике и технологиях", "binary_labels": [0, 1, 0, 1, 0, 1]},
            {"text": "Пример текста 3 о культуре и образовании", "binary_labels": [1, 1, 0, 0, 1, 1]},
            {"text": "Пример текста 4 о здоровье и науке", "binary_labels": [0, 0, 1, 1, 0, 0]},
            {"text": "Пример текста 5 о бизнесе и финансах", "binary_labels": [1, 0, 0, 1, 1, 0]},
        ]
        val_data = [
            {"text": "Валидационный текст 1", "binary_labels": [1, 0, 1, 0, 0, 1]},
            {"text": "Валидационный текст 2", "binary_labels": [0, 1, 0, 1, 1, 0]},
        ]
        test_data = [
            {"text": "Тестовый текст 1", "binary_labels": [1, 0, 0, 1, 0, 1]},
            {"text": "Тестовый текст 2", "binary_labels": [0, 1, 1, 0, 1, 0]},
        ]
        print(f"📊 Тестовые данные: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Создание и обучение классификатора
    print("\n" + "=" * 50)
    classifier = MultiLabelTextClassifier(
        n_iter=10,  # Уменьшено для быстрого тестирования
        max_training_time=30  # 30 секунд для тестирования
    )

    # Обучение
    try:
        classifier.train(train_data, val_data)
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        return

    # Информация о модели
    print("\n" + "=" * 50)
    print("📋 ИНФОРМАЦИЯ О МОДЕЛИ:")
    model_info = classifier.get_model_info()
    for key, value in model_info.items():
        if key == 'parameters':
            print(f"   {key}:")
            for param_key, param_value in value.items():
                print(f"     {param_key}: {param_value}")
        elif key == 'labels':
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value}")

    # Оценка на тестовых данных
    print("\n" + "=" * 50)
    print("🧪 ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ")
    try:
        results = classifier.evaluate(test_data)
        print(f"\n✅ Оценка завершена!")
    except Exception as e:
        print(f"❌ Ошибка при оценке: {e}")

    # Построение графиков
    print("\n" + "=" * 50)
    print("📈 ПОСТРОЕНИЕ ГРАФИКОВ")

    try:
        # Графики обучения
        classifier.plot_training_history(save_path='training_history.png')
    except Exception as e:
        print(f"❌ Ошибка при построении графиков обучения: {e}")

    # Матрицы ошибок (если классов не слишком много)
    try:
        if classifier.label_names is not None and len(classifier.label_names) <= 15:
            classifier.plot_confusion_matrices(test_data, save_path='confusion_matrices.png')
        elif classifier.label_names is not None:
            print(f"⚠️ Слишком много классов ({len(classifier.label_names)}). Пропускаем визуализацию матриц.")
    except Exception as e:
        print(f"❌ Ошибка при построении матриц ошибок: {e}")

    # Сохранение модели
    try:
        classifier.save_model("multilabel_automl_classifier.pkl")
        print(f"✅ Модель сохранена")
    except Exception as e:
        print(f"❌ Ошибка при сохранении модели: {e}")

    # Пример предсказания
    print("\n" + "=" * 50)
    print("🔮 ПРИМЕР ПРЕДСКАЗАНИЯ")

    try:
        if test_data:
            sample_text = test_data[0]['text']
            result = classifier.predict_single(sample_text)

            print(f"\nТекст: {result['text']}")
            print(f"Истинные метки: {test_data[0]['binary_labels']}")
            print(f"Предсказанные метки: {result['predicted_labels']}")
            print(f"Модель: {result['model_name']}")

            if 'probabilities' in result:
                print(f"\nВероятности по классам:")
                for class_name, prob in result['probabilities'].items():
                    print(f"  {class_name}: {prob:.3f}")
    except Exception as e:
        print(f"❌ Ошибка при предсказании: {e}")


if __name__ == "__main__":
    main()