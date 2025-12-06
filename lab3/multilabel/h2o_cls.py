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
    multilabel_confusion_matrix,
    make_scorer
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


# ============================================================================
# КЛАССИЧЕСКИЙ ПОДХОД С RANDOMIZEDSEARCHCV
# ============================================================================

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

            self.best_model_name = max(
                best_models.keys(),
                key=lambda x: best_models[x][score_key] if best_models[x][score_key] is not None else 0
            )

            self.best_model = best_models[self.best_model_name]['model']
            self.best_score = best_models[self.best_model_name][score_key]
            self.best_params = best_models[self.best_model_name]['params']

            print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {self.best_model_name}")
            print(f"   F1 Score: {self.best_score:.3f}")

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

    def get_model_info(self):
        """
        Возвращает информацию о модели
        """
        if not self.is_trained:
            return {"error": "Модель не обучена"}

        label_names_list = self.label_names if self.label_names is not None else []

        return {
            'model_name': self.best_model_name,
            'model_type': type(self.best_model).__name__,
            'best_score': self.best_score,
            'feature_count': len(self.vectorizer.get_feature_names_out()),
            'label_count': len(label_names_list),
        }

    def save_model(self, filename):
        """
        Сохранение модели
        """
        label_names_to_save = list(self.label_names) if self.label_names is not None else None

        joblib.dump({
            'best_model': self.best_model,
            'vectorizer': self.vectorizer,
            'label_binarizer': self.label_binarizer,
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'label_names': label_names_to_save,
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

        loaded_label_names = loaded.get('label_names', None)
        if loaded_label_names is not None:
            self.label_names = [str(cls) for cls in loaded_label_names]
        else:
            self.label_names = None

        self.history = loaded.get('history', {})
        self.is_trained = True
        print(f"📥 Multi-label модель загружена: {filename}")


# ============================================================================
# H2O.AI AUTOML ПОДХОД (ОПТИМИЗИРОВАННЫЙ)
# ============================================================================

try:
    import h2o
    from h2o.automl import H2OAutoML

    H2O_AVAILABLE = True
    print("✅ H2O.ai доступен")
except ImportError:
    print("⚠️ H2O.ai не установлен. Установите: pip install h2o")
    H2O_AVAILABLE = False


class OptimizedH2OAutoMLClassifier:
    """
    Оптимизированный H2O AutoML классификатор для быстрого обучения
    """

    def __init__(self, max_runtime_secs=180, seed=42, n_classes_limit=5):
        """
        Args:
            max_runtime_secs: максимальное время обучения в секундах
            seed: для воспроизводимости
            n_classes_limit: ограничение количества классов для обучения
        """
        if not H2O_AVAILABLE:
            raise ImportError("H2O.ai не установлен. Установите: pip install tpot")

        # Оптимизированный векторizer с меньшим количеством признаков
        self.vectorizer = TfidfVectorizer(
            max_features=500,  # Меньше признаков для скорости
            min_df=1,  # Более либеральные настройки
            max_df=0.95,
            ngram_range=(1, 1),  # Только униграммы
            stop_words=None
        )

        self.label_binarizer = MultiLabelBinarizer()
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed
        self.n_classes_limit = n_classes_limit
        self.is_trained = False
        self.label_names = None
        self.models = {}
        self.h2o_initialized = False

        print(f"⚡ Optimized H2O AutoML Classifier инициализирован:")
        print(f"   Максимальное время: {max_runtime_secs} сек")
        print(f"   Максимальное количество классов: {n_classes_limit}")
        print(f"   Признаков: 500")

    def _init_h2o(self):
        """Инициализация H2O с оптимизированными настройками"""
        if not self.h2o_initialized:
            try:
                # Оптимизированные настройки H2O
                h2o.init(
                    min_mem_size="1G",  # Минимальная память
                    max_mem_size="2G",  # Максимальная память
                    nthreads=2,  # Ограничиваем потоки
                    verbose=False,
                    enable_assertions=False  # Отключаем проверки для скорости
                )
                self.h2o_initialized = True
                print("   ✅ H2O инициализирован с оптимизированными настройками")
            except Exception as e:
                print(f"   ⚠️ Не удалось инициализировать H2O: {e}")
                # Пробуем без специфичных настроек
                h2o.init(verbose=False)
                self.h2o_initialized = True

    def prepare_data(self, data):
        """
        Подготовка данных
        """
        texts = [item['text'] for item in data]
        labels = [item['binary_labels'] for item in data]

        if not hasattr(self.label_binarizer, 'classes_'):
            labels_binary = self.label_binarizer.fit_transform(labels)
            self.label_names = [str(cls) for cls in self.label_binarizer.classes_]
            print(f"   Всего классов: {len(self.label_names)}")
        else:
            labels_binary = self.label_binarizer.transform(labels)

        return texts, labels_binary

    def train(self, train_data, val_data=None):
        """
        Быстрое обучение с оптимизациями
        """
        print("⚡ БЫСТРОЕ ОБУЧЕНИЕ H2O AUTOML С ОПТИМИЗАЦИЯМИ...")

        # Инициализация H2O
        self._init_h2o()

        # Подготовка данных
        X_train, y_train = self.prepare_data(train_data)
        X_train_vec = self.vectorizer.fit_transform(X_train)

        print(f"   Размерность данных: {X_train_vec.shape}")
        print(f"   Примеров: {len(X_train)}")

        # Ограничиваем количество классов для обучения
        if len(self.label_names) > self.n_classes_limit:
            print(f"   ⚠️ Слишком много классов ({len(self.label_names)}).")
            print(f"   Обучаем только первые {self.n_classes_limit} классов...")
            classes_to_train = self.label_names[:self.n_classes_limit]
        else:
            classes_to_train = self.label_names

        print(f"   Классов для обучения: {len(classes_to_train)}")

        # Подготавливаем данные один раз
        feature_names = self.vectorizer.get_feature_names_out()
        X_df = pd.DataFrame(X_train_vec.toarray(), columns=feature_names)

        # Добавляем все метки
        for i, label_name in enumerate(self.label_names):
            if i < len(y_train[0]):  # Проверка на границы
                X_df[label_name] = y_train[:, i]

        # Преобразуем в H2O Frame
        h2o_df = h2o.H2OFrame(X_df)

        # Обучаем модели для каждого выбранного класса
        models_trained = 0
        for i, label_name in enumerate(classes_to_train):
            if i >= len(self.label_names):
                break

            print(f"\n   🏷️  Класс {i + 1}/{len(classes_to_train)}: {label_name}")

            try:
                # Оптимизированные настройки AutoML для скорости
                automl = H2OAutoML(
                    max_runtime_secs=max(30, self.max_runtime_secs // len(classes_to_train)),
                    max_models=2,  # Всего 2 модели
                    seed=self.seed,
                    nfolds=2,  # Только 2 фолда
                    stopping_metric='AUC',
                    sort_metric='AUC',
                    verbosity='error',  # Минимальный вывод
                    exclude_algos=["DeepLearning", "StackedEnsemble"],  # Исключаем медленные алгоритмы
                    include_algos=["GLM", "GBM", "DRF"]  # Только быстрые алгоритмы
                )

                print(f"      🏃 Запуск AutoML (макс. {max(30, self.max_runtime_secs // len(classes_to_train))} сек)...")

                # Запускаем AutoML
                automl.train(
                    x=list(feature_names),
                    y=label_name,
                    training_frame=h2o_df
                )

                # Проверяем, что модель обучена
                if automl.leader is not None:
                    self.models[label_name] = automl.leader
                    models_trained += 1

                    # Быстрая оценка
                    try:
                        lb = automl.leaderboard
                        if lb is not None and len(lb) > 0:
                            print(f"      ✅ Модель обучена: {automl.leader.model_id}")
                            print(f"      📊 AUC: {automl.leader.auc():.3f}")
                    except:
                        print(f"      ✅ Модель обучена")
                else:
                    print(f"      ⚠️ AutoML не создал модель для класса {label_name}")

            except Exception as e:
                print(f"      ❌ Ошибка для класса {label_name}: {str(e)[:100]}...")
                # Пробуем простую GLM модель
                try:
                    print(f"      🔧 Пробуем простую GLM модель...")
                    from h2o.estimators.glm import H2OGeneralizedLinearEstimator
                    glm = H2OGeneralizedLinearEstimator(
                        family="binomial",
                        seed=self.seed,
                        lambda_search=True,
                        alpha=0.5
                    )
                    glm.train(
                        x=list(feature_names),
                        y=label_name,
                        training_frame=h2o_df
                    )
                    self.models[label_name] = glm
                    models_trained += 1
                    print(f"      ✅ GLM модель обучена")
                except Exception as e2:
                    print(f"      ❌ Не удалось обучить даже GLM: {str(e2)[:100]}...")

        if models_trained > 0:
            self.is_trained = True
            print(f"\n✅ Обучение завершено!")
            print(f"   Успешно обучено моделей: {models_trained}/{len(classes_to_train)}")
        else:
            print(f"\n❌ Не удалось обучить ни одну модель!")
            raise Exception("H2O AutoML не смог обучить модели")

    def predict(self, texts, threshold=0.5):
        """
        Предсказание
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()

        if X_vec.shape[1] != len(feature_names):
            # Если размерности не совпадают, переобучаем векторizer
            print("⚠️ Размерности признаков не совпадают, используем fallback...")
            return self._predict_fallback(texts, threshold)

        X_df = pd.DataFrame(X_vec.toarray(), columns=feature_names)

        # Преобразуем в H2O Frame
        try:
            h2o_df = h2o.H2OFrame(X_df)
        except:
            print("⚠️ Ошибка при создании H2O Frame, используем fallback...")
            return self._predict_fallback(texts, threshold)

        predictions_matrix = []

        for label_name in self.label_names:
            if label_name in self.models:
                model = self.models[label_name]
                try:
                    preds = model.predict(h2o_df)
                    if 'p1' in preds.columns:
                        probs = preds['p1'].as_data_frame().values.flatten()
                        binary_preds = (probs >= threshold).astype(int)
                    elif 'predict' in preds.columns:
                        # Для некоторых моделей может не быть p1
                        pred_vals = preds['predict'].as_data_frame().values.flatten()
                        binary_preds = pred_vals.astype(int)
                    else:
                        binary_preds = np.zeros(len(texts))

                    predictions_matrix.append(binary_preds)
                except Exception as e:
                    print(f"⚠️ Ошибка предсказания для {label_name}: {str(e)[:50]}")
                    predictions_matrix.append(np.zeros(len(texts)))
            else:
                predictions_matrix.append(np.zeros(len(texts)))

        predictions = np.array(predictions_matrix).T if predictions_matrix else np.array([])

        if len(predictions) > 0:
            predictions_labels = self.label_binarizer.inverse_transform(predictions)
        else:
            predictions_labels = [[] for _ in range(len(texts))]

        return predictions, predictions_labels, None

    def _predict_fallback(self, texts, threshold=0.5):
        """Fallback метод предсказания"""
        print("Используем fallback предсказание...")
        predictions = np.zeros((len(texts), len(self.label_names)))
        predictions_labels = [[] for _ in range(len(texts))]
        return predictions, predictions_labels, None

    def evaluate(self, test_data):
        """
        Быстрая оценка
        """
        if not self.is_trained:
            return {'accuracy': 0, 'hamming_loss': 1, 'f1_weighted': 0}

        X_test, y_test = self.prepare_data(test_data)

        # Получаем предсказания только для обученных классов
        predictions, pred_labels, _ = self.predict([item['text'] for item in test_data])

        if len(predictions) == 0:
            return {'accuracy': 0, 'hamming_loss': 1, 'f1_weighted': 0}

        # Оцениваем только по тем классам, для которых есть модели
        trained_indices = [i for i, label_name in enumerate(self.label_names)
                           if label_name in self.models]

        if trained_indices:
            y_test_subset = y_test[:, trained_indices]
            predictions_subset = predictions[:, trained_indices]

            accuracy = accuracy_score(y_test_subset, predictions_subset)
            h_loss = hamming_loss(y_test_subset, predictions_subset)
            f1 = f1_score(y_test_subset, predictions_subset, average='weighted')
        else:
            accuracy = h_loss = f1 = 0.0

        print(f"\n📊 Результаты Optimized H2O AutoML:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Hamming Loss: {h_loss:.3f}")
        print(f"   F1 Weighted: {f1:.3f}")
        print(f"   Обучено классов: {len(self.models)}/{len(self.label_names)}")

        return {'accuracy': accuracy, 'hamming_loss': h_loss, 'f1_weighted': f1}

    def get_model_info(self):
        """
        Информация о модели
        """
        if not self.is_trained:
            return {"error": "Модель не обучена"}

        return {
            'model_type': 'Optimized H2O AutoML',
            'feature_count': len(self.vectorizer.get_feature_names_out()),
            'label_count': len(self.label_names) if self.label_names else 0,
            'models_trained': len(self.models),
            'training_time': f"{self.max_runtime_secs} сек"
        }

    def __del__(self):
        """Закрытие H2O соединения"""
        if H2O_AVAILABLE and self.h2o_initialized:
            try:
                h2o.shutdown(prompt=False)
            except:
                pass


# ============================================================================
# БЫСТРЫЙ H2O AUTOML С GLM (САМЫЙ БЫСТРЫЙ ВАРИАНТ)
# ============================================================================

class FastH2OGLMClassifier:
    """
    Быстрый классификатор на основе H2O GLM (Generalized Linear Model)
    Самый быстрый вариант для демонстрации
    """

    def __init__(self, seed=42):
        """
        Args:
            seed: для воспроизводимости
        """
        if not H2O_AVAILABLE:
            raise ImportError("H2O.ai не установлен. Установите: pip install h2o")

        # Минимальный векторizer для скорости
        self.vectorizer = TfidfVectorizer(
            max_features=200,  # Очень мало признаков для скорости
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 1)
        )

        self.label_binarizer = MultiLabelBinarizer()
        self.seed = seed
        self.is_trained = False
        self.label_names = None
        self.models = {}

        print(f"🚀 Fast H2O GLM Classifier инициализирован:")
        print(f"   Признаков: 200 (для скорости)")

    def prepare_data(self, data):
        """
        Подготовка данных
        """
        texts = [item['text'] for item in data]
        labels = [item['binary_labels'] for item in data]

        if not hasattr(self.label_binarizer, 'classes_'):
            labels_binary = self.label_binarizer.fit_transform(labels)
            self.label_names = [str(cls) for cls in self.label_binarizer.classes_]
        else:
            labels_binary = self.label_binarizer.transform(labels)

        return texts, labels_binary

    def train(self, train_data, val_data=None):
        """
        Сверхбыстрое обучение только GLM моделями
        """
        print("🚀 СВЕРХБЫСТРОЕ ОБУЧЕНИЕ H2O GLM...")

        # Инициализация H2O
        try:
            h2o.init(verbose=False, nthreads=1)
        except:
            h2o.init(verbose=False)

        # Подготовка данных
        X_train, y_train = self.prepare_data(train_data)
        X_train_vec = self.vectorizer.fit_transform(X_train)

        print(f"   Данные: {X_train_vec.shape[0]} примеров, {X_train_vec.shape[1]} признаков")
        print(f"   Классов: {len(self.label_names)}")

        # Ограничиваем количество классов для скорости
        max_classes = min(3, len(self.label_names))
        print(f"   Обучаем первые {max_classes} классов...")

        # Подготавливаем данные
        feature_names = self.vectorizer.get_feature_names_out()
        X_df = pd.DataFrame(X_train_vec.toarray(), columns=feature_names)

        # Добавляем метки
        for i in range(max_classes):
            if i < len(self.label_names):
                X_df[self.label_names[i]] = y_train[:, i]

        # Преобразуем в H2O Frame
        h2o_df = h2o.H2OFrame(X_df)

        # Обучаем GLM для каждого класса
        from h2o.estimators.glm import H2OGeneralizedLinearEstimator

        for i in range(max_classes):
            if i >= len(self.label_names):
                break

            label_name = self.label_names[i]
            print(f"   🏷️  Класс {i + 1}/{max_classes}: {label_name}")

            try:
                # Быстрая GLM модель
                glm = H2OGeneralizedLinearEstimator(
                    family="binomial",
                    seed=self.seed,
                    alpha=0.5,  # ElasticNet
                    lambda_search=True,  # Автоподбор регуляризации
                    nlambdas=5,  # Всего 5 значений lambda
                    max_iterations=50  # Ограничение итераций
                )

                glm.train(
                    x=list(feature_names),
                    y=label_name,
                    training_frame=h2o_df
                )

                self.models[label_name] = glm
                print(f"      ✅ GLM обучена")

            except Exception as e:
                print(f"      ❌ Ошибка: {str(e)[:50]}...")

        self.is_trained = len(self.models) > 0

        if self.is_trained:
            print(f"\n✅ Обучение завершено! Моделей: {len(self.models)}")
        else:
            print(f"\n❌ Не удалось обучить модели")

    def predict(self, texts, threshold=0.5):
        """
        Быстрое предсказание
        """
        if not self.is_trained:
            return np.zeros((len(texts), len(self.label_names))), [[] for _ in range(len(texts))], None

        X_vec = self.vectorizer.transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()

        if X_vec.shape[1] != len(feature_names):
            return np.zeros((len(texts), len(self.label_names))), [[] for _ in range(len(texts))], None

        X_df = pd.DataFrame(X_vec.toarray(), columns=feature_names)
        h2o_df = h2o.H2OFrame(X_df)

        predictions_matrix = []

        for label_name in self.label_names:
            if label_name in self.models:
                try:
                    preds = self.models[label_name].predict(h2o_df)
                    if 'p1' in preds.columns:
                        probs = preds['p1'].as_data_frame().values.flatten()
                        binary_preds = (probs >= threshold).astype(int)
                    else:
                        binary_preds = np.zeros(len(texts))
                except:
                    binary_preds = np.zeros(len(texts))
            else:
                binary_preds = np.zeros(len(texts))

            predictions_matrix.append(binary_preds)

        predictions = np.array(predictions_matrix).T if predictions_matrix else np.array([])

        if len(predictions) > 0:
            predictions_labels = self.label_binarizer.inverse_transform(predictions)
        else:
            predictions_labels = [[] for _ in range(len(texts))]

        return predictions, predictions_labels, None

    def evaluate(self, test_data):
        """
        Быстрая оценка
        """
        if not self.is_trained:
            return {'accuracy': 0, 'hamming_loss': 1, 'f1_weighted': 0}

        X_test, y_test = self.prepare_data(test_data)
        predictions, pred_labels, _ = self.predict([item['text'] for item in test_data])

        if len(predictions) == 0:
            return {'accuracy': 0, 'hamming_loss': 1, 'f1_weighted': 0}

        # Оцениваем только по обученным классам
        trained_indices = [i for i, label_name in enumerate(self.label_names)
                           if label_name in self.models]

        if trained_indices:
            y_test_subset = y_test[:, trained_indices]
            predictions_subset = predictions[:, trained_indices]

            if y_test_subset.size > 0 and predictions_subset.size > 0:
                accuracy = accuracy_score(y_test_subset, predictions_subset)
                h_loss = hamming_loss(y_test_subset, predictions_subset)
                f1 = f1_score(y_test_subset, predictions_subset, average='weighted', zero_division=0)
            else:
                accuracy = h_loss = f1 = 0.0
        else:
            accuracy = h_loss = f1 = 0.0

        print(f"\n📊 Результаты Fast H2O GLM:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Hamming Loss: {h_loss:.3f}")
        print(f"   F1 Weighted: {f1:.3f}")

        return {'accuracy': accuracy, 'hamming_loss': h_loss, 'f1_weighted': f1}


# ============================================================================
# СРАВНЕНИЕ ПОДХОДОВ (ОПТИМИЗИРОВАННОЕ)
# ============================================================================

def compare_automl_approaches_optimized(train_data, val_data, test_data):
    """
    Оптимизированное сравнение AutoML подходов
    """
    print("🔬 ОПТИМИЗИРОВАННОЕ СРАВНЕНИЕ AUTOML ПОДХОДОВ")
    print("=" * 60)

    approaches = {}

    # 1. Классический RandomizedSearchCV (быстрый)
    print("\n🎯 1. RANDOMIZED SEARCH CV (быстрый):")
    try:
        random_search = MultiLabelTextClassifier(
            n_iter=5,  # Минимум итераций
            max_training_time=30,
            random_state=42
        )
        random_search.train(train_data, val_data)
        test_results = random_search.evaluate(test_data)

        approaches['random_search'] = {
            'classifier': random_search,
            'accuracy': test_results['accuracy'],
            'f1_weighted': test_results['f1_weighted'],
            'hamming_loss': test_results['hamming_loss']
        }

        print(f"   ✅ Accuracy: {test_results['accuracy']:.3f}")
        print(f"   ✅ F1 Weighted: {test_results['f1_weighted']:.3f}")
        print(f"   ✅ Hamming Loss: {test_results['hamming_loss']:.3f}")

    except Exception as e:
        print(f"   ❌ Ошибка: {e}")

    # 2. Fast H2O GLM (самый быстрый H2O вариант)
    print("\n🚀 2. FAST H2O GLM (самый быстрый):")
    if H2O_AVAILABLE:
        try:
            h2o_classifier = FastH2OGLMClassifier(seed=42)
            h2o_classifier.train(train_data, val_data)
            test_results = h2o_classifier.evaluate(test_data)

            approaches['h2o_fast_glm'] = {
                'classifier': h2o_classifier,
                'accuracy': test_results['accuracy'],
                'f1_weighted': test_results['f1_weighted'],
                'hamming_loss': test_results['hamming_loss']
            }

            print(f"   ✅ Accuracy: {test_results['accuracy']:.3f}")
            print(f"   ✅ F1 Weighted: {test_results['f1_weighted']:.3f}")
            print(f"   ✅ Hamming Loss: {test_results['hamming_loss']:.3f}")

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    else:
        print("   ⚠️ H2O.ai не установлен. Пропускаем...")

    # 3. Оптимизированный H2O AutoML
    print("\n⚡ 3. OPTIMIZED H2O AUTOML (баланс скорости и качества):")
    if H2O_AVAILABLE:
        try:
            h2o_optimized = OptimizedH2OAutoMLClassifier(
                max_runtime_secs=120,  # Больше времени
                seed=42,
                n_classes_limit=3  # Ограничиваем классы
            )
            h2o_optimized.train(train_data, val_data)
            test_results = h2o_optimized.evaluate(test_data)

            approaches['h2o_optimized'] = {
                'classifier': h2o_optimized,
                'accuracy': test_results['accuracy'],
                'f1_weighted': test_results['f1_weighted'],
                'hamming_loss': test_results['hamming_loss']
            }

            print(f"   ✅ Accuracy: {test_results['accuracy']:.3f}")
            print(f"   ✅ F1 Weighted: {test_results['f1_weighted']:.3f}")
            print(f"   ✅ Hamming Loss: {test_results['hamming_loss']:.3f}")

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    else:
        print("   ⚠️ H2O.ai не установлен. Пропускаем...")

    # Сравнение результатов
    print("\n" + "=" * 60)
    print("📊 ИТОГОВОЕ СРАВНЕНИЕ:")
    print("=" * 60)
    print(f"{'Подход':<25} {'Accuracy':<10} {'F1 Weighted':<12} {'Hamming Loss':<12}")
    print("-" * 60)

    for name, result in sorted(approaches.items(),
                               key=lambda x: x[1]['accuracy'],
                               reverse=True):
        acc = result['accuracy']
        f1 = result['f1_weighted']
        h_loss = result['hamming_loss']

        print(f"{name:<25} {acc:<10.3f} {f1:<12.3f} {h_loss:<12.3f}")

    # Рекомендация
    print("\n" + "=" * 60)
    print("💡 РЕКОМЕНДАЦИЯ:")

    if approaches:
        best_approach = max(approaches.items(), key=lambda x: x[1]['accuracy'])[0]
        print(f"   Лучший подход: {best_approach.upper()}")

        if 'h2o' in best_approach:
            print("   H2O показал хорошие результаты при ограниченных ресурсах")
        else:
            print("   Random Search CV обеспечил лучший баланс точности и скорости")

    return approaches


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ (ОПТИМИЗИРОВАННЫЙ)
# ============================================================================

def main_optimized():
    """
    Оптимизированный пример использования
    """
    # Загрузка данных
    try:
        from util.jsonl_process import read_jsonl_basic

        train_data = read_jsonl_basic('../util/news_multilabel_train_data.jsonl')
        val_data = read_jsonl_basic('../util/news_multilabel_val_data.jsonl')
        test_data = read_jsonl_basic('../util/news_multilabel_test_data.jsonl')

        print(f"📊 Данные загружены:")
        print(f"   Train: {len(train_data)} примеров")
        print(f"   Validation: {len(val_data)} примеров")
        print(f"   Test: {len(test_data)} примеров")

        if train_data:
            print(f"\n📝 Пример записи:")
            print(f"   Text: {train_data[0]['text'][:100]}...")
            print(f"   Labels: {train_data[0]['binary_labels']}")
            print(f"   Количество меток: {len(train_data[0]['binary_labels'])}")

    except Exception as e:
        print(f"⚠️ Не удалось загрузить данные: {e}")
        print("Создаем тестовые данные для демонстрации...")

        # Создаем тестовые данные
        train_data = [
            {"text": "Пример текста 1 о спорте и политике", "binary_labels": [1, 0, 1, 0]},
            {"text": "Пример текста 2 об экономике и технологиях", "binary_labels": [0, 1, 0, 1]},
            {"text": "Пример текста 3 о культуре и образовании", "binary_labels": [1, 1, 0, 0]},
            {"text": "Пример текста 4 о здоровье и науке", "binary_labels": [0, 0, 1, 1]},
        ]
        val_data = [
            {"text": "Валидационный текст 1", "binary_labels": [1, 0, 1, 0]},
            {"text": "Валидационный текст 2", "binary_labels": [0, 1, 0, 1]},
        ]
        test_data = [
            {"text": "Тестовый текст 1", "binary_labels": [1, 0, 0, 1]},
            {"text": "Тестовый текст 2", "binary_labels": [0, 1, 1, 0]},
        ]
        print(f"📊 Тестовые данные созданы (упрощенные):")
        print(f"   Train: {len(train_data)} примеров")
        print(f"   Validation: {len(val_data)} примеров")
        print(f"   Test: {len(test_data)} примеров")

    print("\n" + "=" * 60)

    # Вариант 1: RandomizedSearchCV
    print("\n🎯 ВАРИАНТ 1: RANDOMIZED SEARCH CV")
    print("=" * 40)

    try:
        classifier1 = MultiLabelTextClassifier(
            n_iter=3,  # Минимум для скорости
            max_training_time=20,
            random_state=42
        )

        classifier1.train(train_data, val_data)

        # Сохранение модели
        classifier1.save_model("multilabel_fast_random_search.pkl")

        # Графики обучения
        classifier1.plot_training_history(save_path='fast_random_search_history.png')

        # Оценка
        print("\n🧪 ОЦЕНКА НА ТЕСТЕ:")
        results1 = classifier1.evaluate(test_data)
    except Exception as e:
        print(f"❌ Ошибка: {e}")

    # Вариант 2: Fast H2O GLM
    if H2O_AVAILABLE:
        print("\n🚀 ВАРИАНТ 2: FAST H2O GLM")
        print("=" * 40)

        try:
            classifier2 = FastH2OGLMClassifier(seed=42)
            classifier2.train(train_data, val_data)

            # Оценка
            print("\n🧪 ОЦЕНКА НА ТЕСТЕ:")
            results2 = classifier2.evaluate(test_data)
        except Exception as e:
            print(f"❌ Ошибка: {e}")

    # Сравнение подходов
    print("\n" + "=" * 60)
    print("🔬 СРАВНЕНИЕ ВСЕХ ПОДХОДОВ")
    print("=" * 60)

    approaches = compare_automl_approaches_optimized(train_data, val_data, test_data)

    # Пример предсказания
    print("\n" + "=" * 60)
    print("🔮 ПРИМЕР ПРЕДСКАЗАНИЯ")
    print("=" * 60)

    if test_data and approaches:
        sample_text = test_data[0]['text']

        print(f"\nТекст: {sample_text[:100]}...")
        print(f"Истинные метки: {test_data[0]['binary_labels']}")

        # Используем первый успешный классификатор
        for approach_name, result in approaches.items():
            classifier = result['classifier']
            if classifier:
                try:
                    predictions, pred_labels, probs = classifier.predict([sample_text])
                    print(f"\n📊 {approach_name.upper()} предсказание:")
                    print(f"   Метки: {pred_labels[0]}")
                    break
                except:
                    continue

    # Закрытие H2O
    if H2O_AVAILABLE:
        try:
            h2o.shutdown(prompt=False)
            print("\n✅ H2O соединения закрыты")
        except:
            pass


if __name__ == "__main__":
    main_optimized()