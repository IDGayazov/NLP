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

        # Подробный отчет по классам
        print("\n📈 CLASSIFICATION REPORT (по каждому классу):")

        # Для multilabel классификации выводим отчет для каждого класса отдельно
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
# TPOT ПОДХОД (АВТОМАТИЧЕСКОЕ МАШИННОЕ ОБУЧЕНИЕ)
# ============================================================================

try:
    from tpot import TPOTClassifier

    TPOT_AVAILABLE = True
    print("✅ TPOT доступен")
except ImportError:
    print("⚠️ TPOT не установлен. Установите: pip install tpot")
    TPOT_AVAILABLE = False


class TPOTMultiLabelClassifier:
    """
    AutoML классификатор на основе TPOT для многометочной классификации
    Использует генетическое программирование для поиска оптимального пайплайна
    """

    def __init__(self, max_time_mins=5, generations=5, population_size=20,
                 cv=3, random_state=42, verbosity=2):
        """
        Args:
            max_time_mins: максимальное время обучения в минутах
            generations: количество поколений генетического алгоритма
            population_size: размер популяции
            cv: количество фолдов для кросс-валидации
            random_state: для воспроизводимости
            verbosity: уровень вывода информации (0-3)
        """
        if not TPOT_AVAILABLE:
            raise ImportError("TPOT не установлен. Установите: pip install tpot")

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            stop_words=None
        )

        # Инициализируем MultiLabelBinarizer
        self.label_binarizer = MultiLabelBinarizer()

        # Создаем scorer для TPOT
        # В новых версиях TPOT нужно использовать scoring_func вместо scoring
        f1_weighted_scorer = make_scorer(f1_score, average='weighted')

        # Инициализируем TPOT для многометочной классификации
        # TPOT автоматически поддерживает multilabel через OneVsRestClassifier
        try:
            # Попробуем разные варианты инициализации в зависимости от версии TPOT
            self.tpot = TPOTClassifier(
                generations=generations,
                population_size=population_size,
                cv=cv,
                random_state=random_state,
                verbosity=verbosity,  # Старая версия
                max_time_mins=max_time_mins,
                n_jobs=-1,
                config_dict='TPOT light',  # Более быстрые конфигурации
                template='Selector-Transformer-Classifier'  # Стандартный шаблон
            )
        except TypeError as e:
            if "'scoring'" in str(e) or "'verbosity'" in str(e):
                # Пробуем с новыми параметрами
                print("Используем параметры для новой версии TPOT...")
                self.tpot = TPOTClassifier(
                    generations=generations,
                    population_size=population_size,
                    cv=cv,
                    random_state=random_state,
                    max_time_mins=max_time_mins,
                    n_jobs=-1,
                    config_dict='TPOT light',
                    template='Selector-Transformer-Classifier',
                    scoring=f1_weighted_scorer,  # Новая версия
                    verbosity=verbosity  # Новая версия
                )
            else:
                raise e

        self.max_time_mins = max_time_mins
        self.random_state = random_state
        self.is_trained = False
        self.label_names = None
        self.training_history = []

        print(f"🧬 TPOT Multi-label Classifier инициализирован:")
        print(f"   Максимальное время: {max_time_mins} мин")
        print(f"   Поколений: {generations}")
        print(f"   Размер популяции: {population_size}")
        print(f"   Метрика: F1 weighted")

    def prepare_data(self, data):
        """
        Подготовка данных: извлекаем тексты и метки
        """
        texts = [item['text'] for item in data]
        labels = [item['binary_labels'] for item in data]

        # Преобразуем в бинарный формат
        if not hasattr(self.label_binarizer, 'classes_'):
            labels_binary = self.label_binarizer.fit_transform(labels)
            self.label_names = [str(cls) for cls in self.label_binarizer.classes_]
            print(f"   Количество классов: {len(self.label_names)}")
        else:
            labels_binary = self.label_binarizer.transform(labels)

        return texts, labels_binary

    def train(self, train_data, val_data=None):
        """
        Обучение TPOT классификатора
        """
        print("🧬 ЗАПУСК ГЕНЕТИЧЕСКОГО ПРОГРАММИРОВАНИЯ TPOT...")

        # Подготовка данных
        X_train, y_train = self.prepare_data(train_data)

        # Векторизация текстов
        print("📊 Векторизация текстов...")
        X_train_vec = self.vectorizer.fit_transform(X_train)

        print(f"   Размерность признаков: {X_train_vec.shape}")
        print(f"   Размерность меток: {y_train.shape}")

        # Преобразуем в плотный массив для TPOT
        X_train_dense = X_train_vec.toarray()

        # Обучение TPOT
        print("\n🎯 TPOT ищет оптимальный пайплайн...")
        print(f"   Используется {self.tpot.n_jobs} ядер CPU")

        try:
            self.tpot.fit(X_train_dense, y_train)
        except Exception as e:
            print(f"❌ Ошибка при обучении TPOT: {e}")
            print("Попробуем обучить с меньшим количеством признаков...")

            # Пробуем с меньшим количеством признаков
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                min_df=2,
                max_df=0.9,
                ngram_range=(1, 1),
                stop_words=None
            )

            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_train_dense = X_train_vec.toarray()
            print(f"   Новая размерность признаков: {X_train_vec.shape}")

            self.tpot.fit(X_train_dense, y_train)

        # Оценка на валидации если есть
        if val_data:
            X_val, y_val = self.prepare_data(val_data)
            X_val_vec = self.vectorizer.transform(X_val)
            X_val_dense = X_val_vec.toarray()

            y_val_pred = self.tpot.predict(X_val_dense)
            val_score = f1_score(y_val, y_val_pred, average='weighted')
            val_accuracy = accuracy_score(y_val, y_val_pred)

            print(f"\n✅ Результаты на валидации:")
            print(f"   F1 Weighted: {val_score:.3f}")
            print(f"   Accuracy: {val_accuracy:.3f}")

            # Сохраняем историю
            self.training_history.append({
                'val_f1': val_score,
                'val_accuracy': val_accuracy
            })

        self.is_trained = True

        # Показываем статистику
        print(f"\n📊 СТАТИСТИКА TPOT:")
        try:
            cv_score = self.tpot.score(X_train_dense, y_train)
            print(f"   Оценка на кросс-валидации: {cv_score:.3f}")
        except:
            print(f"   Оценка на кросс-валидации: не доступна")

        try:
            print(f"   Поколений выполнено: {self.tpot.generations_}")
        except:
            pass

        # Показываем лучший пайплайн
        print(f"\n🏆 ЛУЧШИЙ ПАЙПЛАЙН TPOT:")
        try:
            print(self.tpot.fitted_pipeline_)
        except:
            print("Информация о пайплайне не доступна")

        # Экспорт кода лучшего пайплайна
        try:
            export_filename = f'tpot_best_pipeline_{self.random_state}.py'
            self.tpot.export(export_filename)
            print(f"💾 Код лучшего пайплайна сохранен: {export_filename}")
        except Exception as e:
            print(f"⚠️ Не удалось экспортировать пайплайн: {e}")

    def predict(self, texts, threshold=0.5):
        """
        Предсказание для списка текстов
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)
        X_dense = X_vec.toarray()

        # Получаем предсказания
        predictions = self.tpot.predict(X_dense)

        # Для вероятностей (если поддерживается)
        probabilities = None
        if hasattr(self.tpot.fitted_pipeline_, "predict_proba"):
            try:
                probabilities = self.tpot.predict_proba(X_dense)
            except:
                pass

        # Преобразуем обратно в список меток
        predictions_labels = self.label_binarizer.inverse_transform(predictions)

        return predictions, predictions_labels, probabilities

    def evaluate(self, test_data, threshold=0.5):
        """
        Оценка модели на тестовых данных
        """
        X_test, y_test = self.prepare_data(test_data)
        X_test_vec = self.vectorizer.transform(X_test)
        X_test_dense = X_test_vec.toarray()

        # Получаем предсказания
        y_pred = self.tpot.predict(X_test_dense)

        # Вычисляем метрики
        accuracy = accuracy_score(y_test, y_pred)
        h_loss = hamming_loss(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_samples = f1_score(y_test, y_pred, average='samples')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        print("\n📊 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ TPOT:")
        print("-" * 50)
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Hamming Loss: {h_loss:.3f}")
        print(f"Precision (weighted): {precision:.3f}")
        print(f"Recall (weighted): {recall:.3f}")
        print(f"F1 Macro: {f1_macro:.3f}")
        print(f"F1 Weighted: {f1_weighted:.3f}")
        print(f"F1 Samples: {f1_samples:.3f}")

        # Показываем лучший пайплайн
        print(f"\n🏆 ЛУЧШИЙ ПАЙПЛАЙН:")
        try:
            print(self.tpot.fitted_pipeline_)
        except:
            print("Информация о пайплайне не доступна")

        return {
            'accuracy': accuracy,
            'hamming_loss': h_loss,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_samples': f1_samples,
            'precision': precision,
            'recall': recall
        }

    def get_model_info(self):
        """
        Возвращает информацию о модели
        """
        if not self.is_trained:
            return {"error": "Модель не обучена"}

        label_names_list = self.label_names if self.label_names is not None else []

        info = {
            'model_type': 'TPOT AutoML',
            'feature_count': len(self.vectorizer.get_feature_names_out()),
            'label_count': len(label_names_list),
            'training_time': f"{self.max_time_mins} мин"
        }

        try:
            info['best_pipeline'] = str(self.tpot.fitted_pipeline_)
        except:
            info['best_pipeline'] = "Не доступен"

        return info

    def save_model(self, filename):
        """
        Сохранение модели
        """
        label_names_to_save = list(self.label_names) if self.label_names is not None else None

        joblib.dump({
            'tpot': self.tpot,
            'vectorizer': self.vectorizer,
            'label_binarizer': self.label_binarizer,
            'label_names': label_names_to_save,
            'training_history': self.training_history,
            'max_time_mins': self.max_time_mins,
            'random_state': self.random_state
        }, filename, compress=3)
        print(f"💾 TPOT модель сохранена: {filename}")

    def load_model(self, filename):
        """
        Загрузка модели
        """
        loaded = joblib.load(filename)
        self.tpot = loaded['tpot']
        self.vectorizer = loaded['vectorizer']
        self.label_binarizer = loaded['label_binarizer']

        loaded_label_names = loaded.get('label_names', None)
        if loaded_label_names is not None:
            self.label_names = [str(cls) for cls in loaded_label_names]
        else:
            self.label_names = None

        self.training_history = loaded.get('training_history', [])
        self.max_time_mins = loaded.get('max_time_mins', 5)
        self.random_state = loaded.get('random_state', 42)
        self.is_trained = True
        print(f"📥 TPOT модель загружена: {filename}")


# ============================================================================
# УПРОЩЕННЫЙ TPOT ДЛЯ БЫСТРОГО ТЕСТИРОВАНИЯ
# ============================================================================

class SimpleTPOTMultiLabelClassifier:
    """
    Упрощенный TPOT классификатор для быстрого тестирования
    """

    def __init__(self, max_time_mins=2, random_state=42):
        """
        Args:
            max_time_mins: максимальное время обучения в минутах
            random_state: для воспроизводимости
        """
        if not TPOT_AVAILABLE:
            raise ImportError("TPOT не установлен. Установите: pip install tpot")

        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Еще меньше для скорости
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 1)
        )

        self.label_binarizer = MultiLabelBinarizer()

        # Минимальная конфигурация для быстрого тестирования
        # Используем try-except для совместимости с разными версиями TPOT
        try:
            self.tpot = TPOTClassifier(
                generations=1,
                population_size=5,
                cv=2,
                random_state=random_state,
                max_time_mins=max_time_mins,
                n_jobs=1,  # Только 1 ядро для стабильности
                config_dict='TPOT light',
                verbosity=1
            )
        except TypeError:
            # Новая версия TPOT
            accuracy_scorer = make_scorer(accuracy_score)
            self.tpot = TPOTClassifier(
                generations=1,
                population_size=5,
                cv=2,
                random_state=random_state,
                max_time_mins=max_time_mins,
                n_jobs=1,
                config_dict='TPOT light',
                verbosity=1,
                scoring=accuracy_scorer
            )

        self.max_time_mins = max_time_mins
        self.random_state = random_state
        self.is_trained = False
        self.label_names = None

        print(f"⚡ Simple TPOT Classifier инициализирован:")
        print(f"   Максимальное время: {max_time_mins} мин")
        print(f"   Для быстрого тестирования")

    def prepare_data(self, data):
        """
        Подготовка данных
        """
        texts = [item['text'] for item in data]
        labels = [item['binary_labels'] for item in data]

        if not hasattr(self.label_binarizer, 'classes_'):
            labels_binary = self.label_binarizer.fit_transform(labels)
            self.label_names = [str(cls) for cls in self.label_binarizer.classes_]
            print(f"   Классов: {len(self.label_names)}")
        else:
            labels_binary = self.label_binarizer.transform(labels)

        return texts, labels_binary

    def train(self, train_data, val_data=None):
        """
        Быстрое обучение
        """
        print("⚡ БЫСТРОЕ ОБУЧЕНИЕ TPOT...")

        X_train, y_train = self.prepare_data(train_data)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_train_dense = X_train_vec.toarray()

        print(f"   Размерность: {X_train_vec.shape}")
        print(f"   Примеров: {len(X_train)}")
        print(f"   Признаков: {X_train_vec.shape[1]}")
        print(f"   Меток: {y_train.shape[1]}")

        # Проверяем, что данные не слишком большие для быстрого теста
        if X_train_dense.shape[0] > 1000 or X_train_dense.shape[1] > 1000:
            print("   ⚠️ Данные большие, TPOT может работать медленно")
            print("   Уменьшаем размерность...")

            # Уменьшаем размерность
            self.vectorizer = TfidfVectorizer(
                max_features=500,
                min_df=1,
                max_df=0.95,
                ngram_range=(1, 1)
            )
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_train_dense = X_train_vec.toarray()
            print(f"   Новая размерность: {X_train_vec.shape}")

        print("   🏃 Запуск TPOT...")
        self.tpot.fit(X_train_dense, y_train)
        self.is_trained = True

        print(f"\n✅ Обучение завершено")
        try:
            cv_score = self.tpot.score(X_train_dense, y_train)
            print(f"   CV Score: {cv_score:.3f}")
        except:
            print(f"   CV Score: не доступен")

    def predict(self, texts, threshold=0.5):
        """
        Предсказание
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)
        X_dense = X_vec.toarray()

        predictions = self.tpot.predict(X_dense)
        predictions_labels = self.label_binarizer.inverse_transform(predictions)

        return predictions, predictions_labels, None

    def evaluate(self, test_data):
        """
        Быстрая оценка
        """
        X_test, y_test = self.prepare_data(test_data)
        X_test_vec = self.vectorizer.transform(X_test)
        X_test_dense = X_test_vec.toarray()

        y_pred = self.tpot.predict(X_test_dense)

        accuracy = accuracy_score(y_test, y_pred)
        h_loss = hamming_loss(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"\n📊 Результаты Simple TPOT:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Hamming Loss: {h_loss:.3f}")
        print(f"   F1 Weighted: {f1:.3f}")

        return {'accuracy': accuracy, 'hamming_loss': h_loss, 'f1_weighted': f1}


# ============================================================================
# АЛЬТЕРНАТИВНЫЙ ПОДХОД БЕЗ TPOT
# ============================================================================

class LightAutoMLClassifier:
    """
    Легковесный AutoML классификатор без TPOT
    """

    def __init__(self, random_state=42):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2)
        )

        self.label_binarizer = MultiLabelBinarizer()
        self.best_model = None
        self.best_model_name = None
        self.label_names = None
        self.is_trained = False
        self.random_state = random_state

        print(f"⚡ Light AutoML Classifier инициализирован")

    def prepare_data(self, data):
        """
        Подготовка данных
        """
        texts = [item['text'] for item in data]
        labels = [item['binary_labels'] for item in data]

        if not hasattr(self.label_binarizer, 'classes_'):
            labels_binary = self.label_binarizer.fit_transform(labels)
            self.label_names = [str(cls) for cls in self.label_binarizer.classes_]
            print(f"   Классов: {len(self.label_names)}")
        else:
            labels_binary = self.label_binarizer.transform(labels)

        return texts, labels_binary

    def train(self, train_data, val_data=None):
        """
        Быстрое обучение
        """
        print("⚡ БЫСТРОЕ ОБУЧЕНИЕ LIGHT AUTOML...")

        X_train, y_train = self.prepare_data(train_data)
        X_train_vec = self.vectorizer.fit_transform(X_train)

        print(f"   Размерность: {X_train_vec.shape}")

        # Тестируем несколько простых моделей
        models = {
            'logistic': OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=self.random_state)),
            'naive_bayes': OneVsRestClassifier(MultinomialNB()),
            'random_forest': OneVsRestClassifier(
                RandomForestClassifier(n_estimators=50, random_state=self.random_state))
        }

        best_score = 0
        best_model = None
        best_name = None

        # Быстрая оценка на подмножестве данных
        sample_size = min(500, X_train_vec.shape[0])
        if X_train_vec.shape[0] > sample_size:
            indices = np.random.choice(X_train_vec.shape[0], sample_size, replace=False)
            X_sample = X_train_vec[indices]
            y_sample = y_train[indices]
        else:
            X_sample = X_train_vec
            y_sample = y_train

        print(f"   Тестируем модели на {X_sample.shape[0]} примерах...")

        for name, model in models.items():
            print(f"   🔍 Тестируем {name}...")
            try:
                # Быстрая кросс-валидация
                scores = cross_val_score(model, X_sample, y_sample, cv=3,
                                         scoring='f1_weighted', n_jobs=1)
                score = np.mean(scores)
                print(f"      F1: {score:.3f}")

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = name
            except Exception as e:
                print(f"      ❌ Ошибка: {e}")

        # Обучаем лучшую модель на всех данных
        if best_model:
            print(f"\n🏆 Лучшая модель: {best_name} (F1: {best_score:.3f})")
            print(f"   Обучаем на всех данных...")
            best_model.fit(X_train_vec, y_train)
            self.best_model = best_model
            self.best_model_name = best_name
            self.is_trained = True

            # Оценка на валидации если есть
            if val_data:
                X_val, y_val = self.prepare_data(val_data)
                X_val_vec = self.vectorizer.transform(X_val)
                y_val_pred = self.best_model.predict(X_val_vec)
                val_score = f1_score(y_val, y_val_pred, average='weighted')
                print(f"   F1 на валидации: {val_score:.3f}")
        else:
            raise Exception("Не удалось обучить ни одну модель")

    def predict(self, texts, threshold=0.5):
        """
        Предсказание
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        X_vec = self.vectorizer.transform(texts)

        predictions = self.best_model.predict(X_vec)
        predictions_labels = self.label_binarizer.inverse_transform(predictions)

        return predictions, predictions_labels, None

    def evaluate(self, test_data):
        """
        Оценка
        """
        X_test, y_test = self.prepare_data(test_data)
        X_test_vec = self.vectorizer.transform(X_test)

        y_pred = self.best_model.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)
        h_loss = hamming_loss(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"\n📊 Результаты Light AutoML:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Hamming Loss: {h_loss:.3f}")
        print(f"   F1 Weighted: {f1:.3f}")
        print(f"   Модель: {self.best_model_name}")

        return {'accuracy': accuracy, 'hamming_loss': h_loss, 'f1_weighted': f1}

    def get_model_info(self):
        """
        Информация о модели
        """
        if not self.is_trained:
            return {"error": "Модель не обучена"}

        return {
            'model_type': 'Light AutoML',
            'model_name': self.best_model_name,
            'feature_count': len(self.vectorizer.get_feature_names_out()),
            'label_count': len(self.label_names) if self.label_names else 0
        }


# ============================================================================
# СРАВНЕНИЕ ПОДХОДОВ
# ============================================================================

def compare_automl_approaches(train_data, val_data, test_data,
                              tpot_time_mins=2, random_search_time=60):
    """
    Сравнение разных AutoML подходов для многометочной классификации
    """
    print("🔬 СРАВНЕНИЕ AUTOML ПОДХОДОВ ДЛЯ MULTILABEL")
    print("=" * 60)

    approaches = {}

    # 1. Классический RandomizedSearchCV
    print("\n🎯 1. RANDOMIZED SEARCH CV (OneVsRest + классические модели):")
    try:
        random_search = MultiLabelTextClassifier(
            n_iter=10,
            max_training_time=min(random_search_time, 30),
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

    # 2. Light AutoML (без TPOT)
    print("\n⚡ 2. LIGHT AUTOML (быстрый подбор моделей):")
    try:
        light_automl = LightAutoMLClassifier(random_state=42)
        light_automl.train(train_data, val_data)
        test_results = light_automl.evaluate(test_data)

        approaches['light_automl'] = {
            'classifier': light_automl,
            'accuracy': test_results['accuracy'],
            'f1_weighted': test_results['f1_weighted'],
            'hamming_loss': test_results['hamming_loss']
        }

        print(f"   ✅ Accuracy: {test_results['accuracy']:.3f}")
        print(f"   ✅ F1 Weighted: {test_results['f1_weighted']:.3f}")
        print(f"   ✅ Hamming Loss: {test_results['hamming_loss']:.3f}")

    except Exception as e:
        print(f"   ❌ Ошибка: {e}")

    # 3. TPOT AutoML (если доступен)
    if TPOT_AVAILABLE:
        print("\n🧬 3. SIMPLE TPOT AUTOML (генетическое программирование):")
        try:
            tpot_classifier = SimpleTPOTMultiLabelClassifier(
                max_time_mins=min(tpot_time_mins, 2),
                random_state=42
            )
            tpot_classifier.train(train_data, val_data)
            test_results = tpot_classifier.evaluate(test_data)

            approaches['tpot_simple'] = {
                'classifier': tpot_classifier,
                'accuracy': test_results['accuracy'],
                'f1_weighted': test_results['f1_weighted'],
                'hamming_loss': test_results['hamming_loss']
            }

            print(f"   ✅ Accuracy: {test_results['accuracy']:.3f}")
            print(f"   ✅ F1 Weighted: {test_results['f1_weighted']:.3f}")
            print(f"   ✅ Hamming Loss: {test_results['hamming_loss']:.3f}")

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")

    # 4. Базовый подход
    print("\n📊 4. БАЗОВЫЙ ПОДХОД (LogisticRegression OneVsRest):")
    try:
        from sklearn.linear_model import LogisticRegression

        # Создаем простой пайплайн
        base_vectorizer = TfidfVectorizer(max_features=5000)
        base_model = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))

        # Подготовка данных
        texts_train = [item['text'] for item in train_data]
        labels_train = [item['binary_labels'] for item in train_data]

        label_binarizer = MultiLabelBinarizer()
        y_train_binary = label_binarizer.fit_transform(labels_train)

        # Векторизация и обучение
        X_train_vec = base_vectorizer.fit_transform(texts_train)
        base_model.fit(X_train_vec, y_train_binary)

        # Оценка на тесте
        texts_test = [item['text'] for item in test_data]
        labels_test = [item['binary_labels'] for item in test_data]

        y_test_binary = label_binarizer.transform(labels_test)
        X_test_vec = base_vectorizer.transform(texts_test)
        y_pred = base_model.predict(X_test_vec)

        accuracy = accuracy_score(y_test_binary, y_pred)
        f1 = f1_score(y_test_binary, y_pred, average='weighted')
        h_loss = hamming_loss(y_test_binary, y_pred)

        approaches['baseline'] = {
            'classifier': None,
            'accuracy': accuracy,
            'f1_weighted': f1,
            'hamming_loss': h_loss
        }

        print(f"   ✅ Accuracy: {accuracy:.3f}")
        print(f"   ✅ F1 Weighted: {f1:.3f}")
        print(f"   ✅ Hamming Loss: {h_loss:.3f}")

    except Exception as e:
        print(f"   ❌ Ошибка: {e}")

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

        if 'tpot' in best_approach:
            print("   TPOT нашел наиболее оптимальный пайплайн автоматически")
        elif 'random_search' in best_approach:
            print("   Random Search CV обеспечил лучший баланс точности и времени")
        elif 'light_automl' in best_approach:
            print("   Light AutoML показал хорошие результаты при быстром обучении")
        else:
            print("   Базовый подход оказался достаточным для ваших данных")

    return approaches


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

def main():
    """
    Пример использования многометочных классификаторов
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

        # Пример первой записи
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
            {"text": "Пример текста 1 о спорте и политике", "binary_labels": [1, 0, 1, 0, 1, 0]},
            {"text": "Пример текста 2 об экономике и технологиях", "binary_labels": [0, 1, 0, 1, 0, 1]},
            {"text": "Пример текста 3 о культуре и образовании", "binary_labels": [1, 1, 0, 0, 1, 1]},
            {"text": "Пример текста 4 о здоровье и науке", "binary_labels": [0, 0, 1, 1, 0, 0]},
            {"text": "Пример текста 5 о бизнесе и финансах", "binary_labels": [1, 0, 0, 1, 1, 0]},
            {"text": "Пример текста 6 о спортивных событиях", "binary_labels": [1, 0, 0, 0, 0, 0]},
            {"text": "Пример текста 7 о политических решениях", "binary_labels": [0, 1, 0, 0, 0, 0]},
            {"text": "Пример текста 8 о технологических инновациях", "binary_labels": [0, 0, 1, 0, 0, 0]},
        ]
        val_data = [
            {"text": "Валидационный текст 1", "binary_labels": [1, 0, 1, 0, 0, 1]},
            {"text": "Валидационный текст 2", "binary_labels": [0, 1, 0, 1, 1, 0]},
        ]
        test_data = [
            {"text": "Тестовый текст 1", "binary_labels": [1, 0, 0, 1, 0, 1]},
            {"text": "Тестовый текст 2", "binary_labels": [0, 1, 1, 0, 1, 0]},
            {"text": "Тестовый текст 3", "binary_labels": [1, 1, 0, 1, 0, 0]},
        ]
        print(f"📊 Тестовые данные созданы:")
        print(f"   Train: {len(train_data)} примеров")
        print(f"   Validation: {len(val_data)} примеров")
        print(f"   Test: {len(test_data)} примеров")

    print("\n" + "=" * 60)

    # Вариант 1: Использование RandomizedSearchCV подхода
    print("\n🎯 ВАРИАНТ 1: RANDOMIZED SEARCH CV")
    print("=" * 40)

    classifier1 = MultiLabelTextClassifier(
        n_iter=8,  # Быстрый тест
        max_training_time=20,  # 20 секунд
        random_state=42
    )

    classifier1.train(train_data, val_data)

    # Сохранение модели
    classifier1.save_model("multilabel_random_search.pkl")

    # Графики обучения
    classifier1.plot_training_history(save_path='random_search_history.png')

    # Оценка
    print("\n🧪 ОЦЕНКА НА ТЕСТЕ:")
    results1 = classifier1.evaluate(test_data)

    # Вариант 2: Использование Light AutoML
    print("\n⚡ ВАРИАНТ 2: LIGHT AUTOML")
    print("=" * 40)

    classifier2 = LightAutoMLClassifier(random_state=42)
    classifier2.train(train_data, val_data)

    # Оценка
    print("\n🧪 ОЦЕНКА НА ТЕСТЕ:")
    results2 = classifier2.evaluate(test_data)

    # Вариант 3: Использование TPOT (если доступен)
    if TPOT_AVAILABLE:
        print("\n🧬 ВАРИАНТ 3: SIMPLE TPOT")
        print("=" * 40)

        try:
            classifier3 = SimpleTPOTMultiLabelClassifier(
                max_time_mins=1,  # 1 минута для быстрого теста
                random_state=42
            )

            classifier3.train(train_data, val_data)

            # Оценка
            print("\n🧪 ОЦЕНКА НА ТЕСТЕ:")
            results3 = classifier3.evaluate(test_data)
        except Exception as e:
            print(f"❌ Ошибка с TPOT: {e}")

    # Сравнение подходов
    print("\n" + "=" * 60)
    print("🔬 СРАВНЕНИЕ ВСЕХ ПОДХОДОВ")
    print("=" * 60)

    approaches = compare_automl_approaches(
        train_data,
        val_data,
        test_data,
        tpot_time_mins=1,
        random_search_time=20
    )

    # Пример предсказания
    print("\n" + "=" * 60)
    print("🔮 ПРИМЕР ПРЕДСКАЗАНИЯ")
    print("=" * 60)

    if test_data and approaches:
        sample_text = test_data[0]['text']

        print(f"\nТекст: {sample_text[:100]}...")
        print(f"Истинные метки: {test_data[0]['binary_labels']}")

        # Предсказание с лучшей моделью
        best_approach_name = max(approaches.items(), key=lambda x: x[1]['accuracy'])[0]
        best_classifier = approaches[best_approach_name]['classifier']

        if best_classifier:
            predictions, pred_labels, probs = best_classifier.predict([sample_text])
            print(f"\n📊 {best_approach_name.upper()} предсказание:")
            print(f"   Метки: {pred_labels[0]}")

            # Вывод информации о модели
            print(f"\n📋 ИНФОРМАЦИЯ О ЛУЧШЕЙ МОДЕЛИ:")
            model_info = best_classifier.get_model_info()
            for key, value in model_info.items():
                if key not in ['best_pipeline', 'parameters']:
                    print(f"   {key}: {value}")


if __name__ == "__main__":
    main()