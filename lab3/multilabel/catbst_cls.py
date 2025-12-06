from catboost import CatBoostClassifier, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, multilabel_confusion_matrix, hamming_loss
from sklearn.dummy import DummyClassifier
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings

warnings.filterwarnings('ignore')


class MultiLabelCatBoostClassifier:
    """
    Многометочный классификатор на основе градиентного бустинга CatBoost
    """

    def __init__(self, n_labels=14, iterations=500, depth=6, learning_rate=0.1,
                 random_state=42, task_type="CPU", verbose=100):
        """
        Args:
            n_labels: количество меток/классов
            iterations: количество итераций бустинга
            depth: глубина деревьев
            learning_rate: скорость обучения
            random_state: seed для воспроизводимости
            task_type: "CPU" или "GPU" (если доступно)
            verbose: частота вывода логов (0 - без вывода)
        """
        self.vectorizer = None  # Инициализируем позже
        self.n_labels = n_labels
        self.is_trained = False
        self.estimators_ = []  # будем хранить классификаторы для каждой метки отдельно
        self.single_class_labels = set()  # метки с только одним классом
        self.loss_history = []  # история потерь
        self.val_loss_history = []  # история потерь на валидации
        self.accuracy_history = []  # история точности
        self.val_accuracy_history = []  # история точности на валидации
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.task_type = task_type
        self.verbose = verbose
        self.feature_importances_ = None  # важность признаков
        self.eval_history = []  # история обучения

    def _create_catboost_classifier(self):
        """Создает CatBoost классификатор"""
        return CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            random_seed=self.random_state,
            task_type=self.task_type,
            verbose=self.verbose,
            loss_function='Logloss',
            eval_metric='Accuracy',
            early_stopping_rounds=50,
            use_best_model=True,
            od_type='Iter',
            l2_leaf_reg=3,
            border_count=254
        )

    def prepare_data(self, data):
        """
        Подготовка данных: извлекаем тексты и многометочные метки
        """
        texts = [item['text'] for item in data]
        labels = [item['binary_labels'] for item in data]
        return texts, np.array(labels)

    def train(self, train_data, val_data=None):
        """
        Обучение модели с отслеживанием метрик
        """
        print("🎯 ОБУЧЕНИЕ МНОГОМЕТОЧНОГО CATBOOST КЛАССИФИКАТОРА...")

        # Подготовка данных
        X_train, y_train = self.prepare_data(train_data)

        print(f"📊 Размер тренировочных данных: {len(X_train)}")
        print(f"📊 Количество меток: {self.n_labels}")
        print(f"📊 Формат меток: {y_train.shape}")

        # Инициализация и обучение векторизатора
        print("📊 Векторизация текстов...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        # Конвертируем в dense формат для CatBoost
        X_train_dense = X_train_vec.toarray()
        print(f"   Размерность признаков: {X_train_dense.shape}")

        # Подготовка валидационных данных, если есть
        eval_set = None
        if val_data:
            X_val, y_val = self.prepare_data(val_data)
            X_val_vec = self.vectorizer.transform(X_val)
            X_val_dense = X_val_vec.toarray()
            eval_set = [(X_val_dense, y_val)]

        # Обучаем отдельный классификатор для каждой метки
        print("🐱 Обучение CatBoost для каждой метки...")

        # Сброс списков
        self.estimators_ = []
        self.single_class_labels = set()
        self.feature_importances_ = np.zeros((self.n_labels, X_train_dense.shape[1]))
        self.eval_history = []

        for label_idx in range(self.n_labels):
            y_single = y_train[:, label_idx]
            unique_classes = np.unique(y_single)

            if len(unique_classes) < 2:
                # Если только один класс, используем DummyClassifier
                print(f"   Метка {label_idx}: только один класс ({unique_classes[0]}) - используем DummyClassifier")
                clf = DummyClassifier(strategy='constant', constant=unique_classes[0])
                clf.fit(X_train_dense, y_single)
                self.single_class_labels.add(label_idx)
            else:
                # Если два класса, используем CatBoost
                print(f"   Метка {label_idx}: обучение CatBoost...")
                clf = self._create_catboost_classifier()

                # Подготовка eval_set для конкретной метки
                if eval_set:
                    X_val_dense, y_val = eval_set[0]
                    y_val_single = y_val[:, label_idx]
                    val_pool = Pool(X_val_dense, label=y_val_single)
                    train_pool = Pool(X_train_dense, label=y_single)

                    clf.fit(
                        train_pool,
                        eval_set=val_pool,
                        verbose_eval=self.verbose,
                        plot=False
                    )
                else:
                    train_pool = Pool(X_train_dense, label=y_single)
                    clf.fit(train_pool, verbose_eval=self.verbose, plot=False)

                # Сохраняем историю обучения
                if hasattr(clf, 'get_evals_result'):
                    history = clf.get_evals_result()
                    if history:
                        self.eval_history.append(history)

                # Сохраняем важность признаков
                if hasattr(clf, 'get_feature_importance'):
                    self.feature_importances_[label_idx] = clf.get_feature_importance()

            self.estimators_.append(clf)

            # Прогресс
            if (label_idx + 1) % 5 == 0 or (label_idx + 1) == self.n_labels:
                print(f"   Прогресс: {label_idx + 1}/{self.n_labels} меток обучено")

        self.is_trained = True

        # Оценка на тренировочных данных
        y_pred_train = self._predict_from_estimators(X_train_dense)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_hamming = hamming_loss(y_train, y_pred_train)

        print(f"\n✅ Точность на train: {train_accuracy:.4f}")
        print(f"✅ Потеря Хэмминга на train: {train_hamming:.4f}")
        print(f"✅ Метки с одним классом: {sorted(self.single_class_labels)}")

        self.loss_history.append(train_hamming)
        self.accuracy_history.append(train_accuracy)

        # Оценка на валидации, если есть
        if val_data:
            val_accuracy, val_hamming = self.evaluate(val_data, verbose=False)
            print(f"✅ Точность на val: {val_accuracy:.4f}")
            print(f"✅ Потеря Хэмминга на val: {val_hamming:.4f}")
            self.val_loss_history.append(val_hamming)
            self.val_accuracy_history.append(val_accuracy)

        # Покажем важные признаки для меток с двумя классами
        self._show_important_features(top_n=10)

        # Анализ обученной модели
        self._analyze_model()

    def _predict_from_estimators(self, X_dense):
        """Предсказание с использованием всех обученных классификаторов"""
        predictions = []
        for clf in self.estimators_:
            if hasattr(clf, 'predict'):
                pred = clf.predict(X_dense)
                predictions.append(pred)
            else:
                # Для DummyClassifier
                pred = clf.predict(X_dense)
                predictions.append(pred)
        return np.array(predictions).T

    def _predict_proba_from_estimators(self, X_dense):
        """Предсказание вероятностей с использованием всех обученных классификаторов"""
        probabilities = []
        for idx, clf in enumerate(self.estimators_):
            if idx in self.single_class_labels:
                # Для DummyClassifier просто возвращаем [1, 0] или [0, 1]
                pred = clf.predict(X_dense)
                prob = np.zeros((len(pred), 2))
                for i in range(len(pred)):
                    if pred[i] == 1:
                        prob[i] = [0, 1]  # [P(0), P(1)]
                    else:
                        prob[i] = [1, 0]  # [P(0), P(1)]
                probabilities.append(prob[:, 1])
            else:
                # CatBoost возвращает вероятности
                prob = clf.predict_proba(X_dense)
                probabilities.append(prob[:, 1])  # вероятность класса 1
        return np.array(probabilities).T

    def predict(self, texts):
        """
        Предсказание для списка текстов
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

        if self.vectorizer is None:
            raise Exception("Векторизатор не обучен!")

        X_vec = self.vectorizer.transform(texts)
        X_dense = X_vec.toarray()
        predictions = self._predict_from_estimators(X_dense)
        probabilities = self._predict_proba_from_estimators(X_dense)

        return predictions, probabilities

    def predict_single(self, text, threshold=0.5):
        """
        Предсказание для одного текста с детальной информацией
        """
        predictions, probabilities = self.predict([text])
        pred = predictions[0]
        prob = probabilities[0]

        # Применяем порог для получения бинарных предсказаний
        binary_pred = (prob > threshold).astype(int)

        # Получаем индексы активных меток
        active_labels = [i for i, val in enumerate(binary_pred) if val == 1]

        return {
            'prediction': pred.tolist(),
            'binary_prediction': binary_pred.tolist(),
            'probabilities': prob.tolist(),
            'active_labels': active_labels,
            'confidence': np.mean(prob) if len(prob) > 0 else 0
        }

    def evaluate(self, test_data, verbose=True):
        """
        Оценка модели на тестовых данных
        """
        if self.vectorizer is None:
            raise Exception("Векторизатор не обучен! Сначала обучите модель.")

        X_test, y_test = self.prepare_data(test_data)
        X_test_vec = self.vectorizer.transform(X_test)
        X_test_dense = X_test_vec.toarray()

        y_pred = self._predict_from_estimators(X_test_dense)
        accuracy = accuracy_score(y_test, y_pred)
        hamming = hamming_loss(y_test, y_pred)

        if verbose:
            print(f"\n📊 ОЦЕНКА МОДЕЛИ CATBOOST:")
            print(f"   Точность: {accuracy:.4f}")
            print(f"   Потеря Хэмминга: {hamming:.4f}")

            # Анализ меток с одним классом
            if self.single_class_labels:
                print(f"\n⚠️  Метки с одним классом в тренировочных данных: {sorted(self.single_class_labels)}")
                print("   (для этих меток использовался DummyClassifier)")

            print("\n📊 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО КЛАССАМ:")

            # Отчет по каждому классу
            for i in range(self.n_labels):
                y_true = y_test[:, i]
                y_pred_single = y_pred[:, i]
                unique_classes = np.unique(y_true)

                if len(unique_classes) < 2:
                    print(f"\n   Класс {i} (один класс в тестовых данных: {unique_classes[0]}):")
                    print(f"      Все предсказания: {np.unique(y_pred_single)[0]}")
                    print(f"      Accuracy: {np.mean(y_true == y_pred_single):.4f}")
                elif i in self.single_class_labels:
                    print(f"\n   Класс {i} (один класс в тренировочных данных):")
                    print(f"      Все предсказания: {np.unique(y_pred_single)[0]}")
                    print(f"      Accuracy: {np.mean(y_true == y_pred_single):.4f}")
                else:
                    print(f"\n   Класс {i}:")
                    try:
                        print(classification_report(y_true, y_pred_single,
                                                    target_names=[f'Отсутствует({i})', f'Присутствует({i})'],
                                                    zero_division=0, digits=4))
                    except:
                        print("      Не удалось сгенерировать classification_report")

            # Матрицы ошибок для классов с двумя классами в тестовых данных
            print("\n📈 МАТРИЦЫ ОШИБОК ПО КЛАССАМ (только для меток с двумя классами в тестовых данных):")
            valid_labels = []
            for i in range(self.n_labels):
                if len(np.unique(y_test[:, i])) >= 2 and i not in self.single_class_labels:
                    valid_labels.append(i)

            if valid_labels:
                y_test_valid = y_test[:, valid_labels]
                y_pred_valid = y_pred[:, valid_labels]
                try:
                    cm = multilabel_confusion_matrix(y_test_valid, y_pred_valid)

                    for idx, label_idx in enumerate(valid_labels[:5]):  # Показываем только первые 5
                        print(f"\n   Класс {label_idx}:")
                        print(f"               Предсказано 0  Предсказано 1")
                        print(f"   Реально 0:     {cm[idx][0][0]:^10}        {cm[idx][0][1]:^10}")
                        print(f"   Реально 1:     {cm[idx][1][0]:^10}        {cm[idx][1][1]:^10}")
                except:
                    print("   Не удалось построить матрицы ошибок")
            else:
                print("   Нет меток с двумя классами в тестовых данных для построения матриц ошибок")

        return accuracy, hamming

    def _show_important_features(self, top_n=10):
        """
        Показывает самые важные признаки для меток с двумя классами
        """
        if self.vectorizer is None or self.feature_importances_ is None:
            return

        feature_names = self.vectorizer.get_feature_names_out()

        print(f"\n🐱 ТОП-{top_n} ВАЖНЫХ ПРИЗНАКОВ ДЛЯ МЕТОК С ДВУМЯ КЛАССАМИ:")

        for label_idx in range(self.n_labels):
            if label_idx not in self.single_class_labels and np.sum(self.feature_importances_[label_idx]) > 0:

                print(f"\n   КЛАСС {label_idx}:")

                # Получаем индексы самых важных признаков
                importance_scores = self.feature_importances_[label_idx]
                top_indices = np.argsort(importance_scores)[-top_n:][::-1]

                print(f"      Самые важные признаки:")
                for i, idx in enumerate(top_indices[:top_n]):
                    if idx < len(feature_names):
                        print(f"        {i + 1}. {feature_names[idx]}: {importance_scores[idx]:.6f}")

    def _analyze_model(self):
        """
        Анализ обученной модели CatBoost
        """
        print("\n🐱 АНАЛИЗ МОДЕЛИ CATBOOST:")
        print("=" * 50)

        # Собираем статистику по моделям
        tree_counts = []

        for idx, clf in enumerate(self.estimators_):
            if idx not in self.single_class_labels:
                if hasattr(clf, 'tree_count_'):
                    tree_counts.append(clf.tree_count_)

        if tree_counts:
            print(f"   Среднее количество деревьев: {np.mean(tree_counts):.1f}")
            print(f"   Минимальное количество деревьев: {np.min(tree_counts)}")
            print(f"   Максимальное количество деревьев: {np.max(tree_counts)}")
            print(f"   Всего обучено моделей: {len(self.estimators_)}")
            print(f"   Из них CatBoost моделей: {len(tree_counts)}")
        else:
            print("   Нет CatBoost моделей для анализа (только DummyClassifiers)")

    def save_model(self, filename):
        """
        Сохранение модели
        """
        if not self.is_trained:
            print("⚠️  Модель не обучена. Нечего сохранять.")
            return

        # Сохраняем CatBoost модели отдельно
        catboost_models = {}
        for idx, clf in enumerate(self.estimators_):
            if idx not in self.single_class_labels:
                catboost_models[idx] = clf

        joblib.dump({
            'estimators': self.estimators_,
            'vectorizer': self.vectorizer,
            'n_labels': self.n_labels,
            'single_class_labels': self.single_class_labels,
            'feature_importances': self.feature_importances_,
            'loss_history': self.loss_history,
            'val_loss_history': self.val_loss_history,
            'accuracy_history': self.accuracy_history,
            'val_accuracy_history': self.val_accuracy_history,
            'iterations': self.iterations,
            'depth': self.depth,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            'task_type': self.task_type,
            'verbose': self.verbose,
            'eval_history': self.eval_history
        }, filename)
        print(f"💾 Модель CatBoost сохранена: {filename}")

    def load_model(self, filename):
        """
        Загрузка модели
        """
        loaded = joblib.load(filename)
        self.estimators_ = loaded['estimators']
        self.vectorizer = loaded['vectorizer']
        self.n_labels = loaded.get('n_labels', 14)
        self.single_class_labels = loaded.get('single_class_labels', set())
        self.feature_importances_ = loaded.get('feature_importances', None)
        self.loss_history = loaded.get('loss_history', [])
        self.val_loss_history = loaded.get('val_loss_history', [])
        self.accuracy_history = loaded.get('accuracy_history', [])
        self.val_accuracy_history = loaded.get('val_accuracy_history', [])
        self.iterations = loaded.get('iterations', 500)
        self.depth = loaded.get('depth', 6)
        self.learning_rate = loaded.get('learning_rate', 0.1)
        self.random_state = loaded.get('random_state', 42)
        self.task_type = loaded.get('task_type', "CPU")
        self.verbose = loaded.get('verbose', 100)
        self.eval_history = loaded.get('eval_history', [])
        self.is_trained = True
        print(f"📥 Модель CatBoost загружена: {filename}")

    def plot_training_history(self):
        """
        Построение графиков обучения
        """
        if not self.loss_history:
            print("Нет истории обучения для построения графиков")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # График потерь
        axes[0].plot(self.loss_history, label='Train Loss', marker='o')
        if self.val_loss_history:
            axes[0].plot(self.val_loss_history, label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Hamming Loss')
        axes[0].set_title('Loss History')
        axes[0].legend()
        axes[0].grid(True)

        # График точности
        axes[1].plot(self.accuracy_history, label='Train Accuracy', marker='o')
        if self.val_accuracy_history:
            axes[1].plot(self.val_accuracy_history, label='Val Accuracy', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy History')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_feature_importances(self, top_n=20):
        """
        Визуализация важности признаков
        """
        if self.feature_importances_ is None:
            print("Нет данных о важности признаков")
            return

        # Суммируем важность признаков по всем меткам
        total_importance = np.sum(self.feature_importances_, axis=0)

        if np.sum(total_importance) == 0:
            print("Все важности признаков равны нулю")
            return

        feature_names = self.vectorizer.get_feature_names_out()

        # Получаем топ-N самых важных признаков
        top_indices = np.argsort(total_importance)[-top_n:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = total_importance[top_indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(top_features))

        ax.barh(y_pos, top_importances, align='center', color='mediumseagreen')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()  # Самые важные сверху
        ax.set_xlabel('Важность признака')
        ax.set_title(f'Топ-{top_n} самых важных признаков (CatBoost)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_learning_curves(self, max_labels=5):
        """
        Построение кривых обучения для меток
        """
        if not self.eval_history:
            print("Нет истории обучения для построения кривых")
            return

        valid_labels = [i for i in range(self.n_labels)
                        if i not in self.single_class_labels and i < len(self.eval_history)]

        if not valid_labels:
            print("Нет CatBoost моделей с историей обучения")
            return

        n_labels = min(max_labels, len(valid_labels))

        fig, axes = plt.subplots(1, n_labels, figsize=(5 * n_labels, 4))
        if n_labels == 1:
            axes = [axes]

        for idx, label_idx in enumerate(valid_labels[:n_labels]):
            history = self.eval_history[label_idx]

            if 'learn' in history and 'validation' in history:
                learn_metric = history['learn']['Accuracy']
                val_metric = history['validation']['Accuracy']

                axes[idx].plot(learn_metric, label='Train', linewidth=2)
                axes[idx].plot(val_metric, label='Validation', linewidth=2)
                axes[idx].set_xlabel('Итерации')
                axes[idx].set_ylabel('Accuracy')
                axes[idx].set_title(f'Метка {label_idx}')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)

        plt.suptitle('Кривые обучения CatBoost', fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self, test_data, max_classes=4):
        """
        Визуализация матриц ошибок для первых N классов с двумя классами
        """
        if self.vectorizer is None:
            print("⚠️  Векторизатор не обучен. Не могу построить матрицы ошибок.")
            return

        X_test, y_test = self.prepare_data(test_data)
        X_test_vec = self.vectorizer.transform(X_test)
        X_test_dense = X_test_vec.toarray()
        y_pred = self._predict_from_estimators(X_test_dense)

        # Фильтруем метки с двумя классами в тестовых данных
        valid_labels = []
        for i in range(self.n_labels):
            if len(np.unique(y_test[:, i])) >= 2 and i not in self.single_class_labels:
                valid_labels.append(i)

        if not valid_labels:
            print("Нет меток с двумя классами для построения матриц ошибок")
            return

        n_classes = min(max_classes, len(valid_labels))

        if n_classes == 0:
            print("Нет меток с двумя классами для построения матриц ошибок")
            return

        fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))

        if n_classes == 1:
            axes = [axes]

        for i in range(n_classes):
            label_idx = valid_labels[i]
            cm = multilabel_confusion_matrix(y_test[:, [label_idx]], y_pred[:, [label_idx]])[0]

            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                        xticklabels=['Pred 0', 'Pred 1'],
                        yticklabels=['True 0', 'True 1'],
                        ax=axes[i])
            axes[i].set_title(f'Confusion Matrix - Class {label_idx}')
            axes[i].set_ylabel('True Label')
            axes[i].set_xlabel('Predicted Label')

        plt.tight_layout()
        plt.show()

    def analyze_label_distribution(self, data):
        """
        Анализ распределения меток в данных
        """
        _, y = self.prepare_data(data)

        print("\n📊 АНАЛИЗ РАСПРЕДЕЛЕНИЯ МЕТОК:")
        print("=" * 50)

        for i in range(self.n_labels):
            unique, counts = np.unique(y[:, i], return_counts=True)
            print(f"Метка {i}:")
            for val, count in zip(unique, counts):
                percentage = count / len(y) * 100
                print(f"  Класс {val}: {count} примеров ({percentage:.1f}%)")
            if len(unique) < 2:
                print(f"  ⚠️  Только один класс!")
            print()

    def compare_hyperparameters(self, train_data, val_data=None,
                                iterations_list=[100, 300, 500],
                                depth_list=[4, 6, 8],
                                learning_rate_list=[0.01, 0.05, 0.1]):
        """
        Сравнение разных гиперпараметров CatBoost
        """
        print("🔬 СРАВНЕНИЕ РАЗНЫХ ГИПЕРПАРАМЕТРОВ CATBOOST:")
        print("=" * 60)

        # Подготовка данных
        X_train, y_train = self.prepare_data(train_data)

        # Создаем и обучаем векторизатор
        vectorizer_temp = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        X_train_vec = vectorizer_temp.fit_transform(X_train)
        X_train_dense = X_train_vec.toarray()

        if val_data:
            X_val, y_val = self.prepare_data(val_data)
            X_val_vec = vectorizer_temp.transform(X_val)
            X_val_dense = X_val_vec.toarray()

        results = []

        # Обучаем только на первой метке для быстрого сравнения
        test_label = 0
        y_single = y_train[:, test_label]

        if len(np.unique(y_single)) < 2:
            print("Первая метка имеет только один класс, выбираем другую...")
            for i in range(1, self.n_labels):
                if len(np.unique(y_train[:, i])) >= 2:
                    test_label = i
                    y_single = y_train[:, test_label]
                    break

        print(f"Сравнение на метке {test_label}")

        for iterations in iterations_list:
            for depth in depth_list:
                for lr in learning_rate_list:
                    print(f"\n📊 iterations={iterations}, depth={depth}, lr={lr}:")

                    # Обучаем CatBoost
                    clf = CatBoostClassifier(
                        iterations=iterations,
                        depth=depth,
                        learning_rate=lr,
                        random_seed=self.random_state,
                        task_type=self.task_type,
                        verbose=0,
                        loss_function='Logloss',
                        eval_metric='Accuracy'
                    )

                    if val_data:
                        y_val_single = y_val[:, test_label]
                        train_pool = Pool(X_train_dense, label=y_single)
                        val_pool = Pool(X_val_dense, label=y_val_single)

                        clf.fit(
                            train_pool,
                            eval_set=val_pool,
                            verbose_eval=False
                        )

                        # Оценка
                        y_pred_train = clf.predict(X_train_dense)
                        train_accuracy = accuracy_score(y_single, y_pred_train)

                        y_pred_val = clf.predict(X_val_dense)
                        val_accuracy = accuracy_score(y_val_single, y_pred_val)

                        print(f"   Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
                    else:
                        train_pool = Pool(X_train_dense, label=y_single)
                        clf.fit(train_pool, verbose_eval=False)

                        y_pred_train = clf.predict(X_train_dense)
                        train_accuracy = accuracy_score(y_single, y_pred_train)
                        print(f"   Train Accuracy: {train_accuracy:.4f}")

                    results.append({
                        'iterations': iterations,
                        'depth': depth,
                        'learning_rate': lr,
                        'train_accuracy': train_accuracy,
                        'val_accuracy': val_accuracy if val_data else None
                    })

        # Визуализация результатов
        self._plot_catboost_hyperparameter_comparison(results)

    def _plot_catboost_hyperparameter_comparison(self, results):
        """Построение графиков сравнения разных гиперпараметров"""
        if not results:
            print("Нет результатов для построения графиков")
            return

        # Создаем сетку для визуализации
        iterations_list = sorted(list(set(r['iterations'] for r in results)))
        depth_list = sorted(list(set(r['depth'] for r in results)))
        lr_list = sorted(list(set(r['learning_rate'] for r in results)))

        # Для каждой learning rate создаем heatmap
        fig, axes = plt.subplots(1, len(lr_list), figsize=(5 * len(lr_list), 4))
        if len(lr_list) == 1:
            axes = [axes]

        for idx, lr in enumerate(lr_list):
            # Фильтруем результаты по learning rate
            lr_results = [r for r in results if r['learning_rate'] == lr]

            # Создаем матрицу для heatmap
            accuracy_matrix = np.zeros((len(depth_list), len(iterations_list)))

            for r in lr_results:
                i = depth_list.index(r['depth'])
                j = iterations_list.index(r['iterations'])
                accuracy_matrix[i, j] = r['train_accuracy']

            # Heatmap
            im = axes[idx].imshow(accuracy_matrix, cmap='YlGn', aspect='auto')

            # Настройки осей
            axes[idx].set_xticks(np.arange(len(iterations_list)))
            axes[idx].set_xticklabels(iterations_list)
            axes[idx].set_yticks(np.arange(len(depth_list)))
            axes[idx].set_yticklabels(depth_list)
            axes[idx].set_xlabel('Iterations')
            axes[idx].set_ylabel('Depth')
            axes[idx].set_title(f'Learning Rate = {lr}')

            # Добавляем значения в ячейки
            for i in range(len(depth_list)):
                for j in range(len(iterations_list)):
                    text = axes[idx].text(j, i, f'{accuracy_matrix[i, j]:.3f}',
                                          ha="center", va="center", color="black")

        plt.suptitle('Сравнение гиперпараметров CatBoost', fontsize=14)
        plt.tight_layout()
        plt.show()


# Пример использования
def main():
    """
    Пример использования многометочного CatBoost классификатора
    """

    # Функция для чтения данных
    def read_jsonl_basic(filepath):
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    # Загрузка данных
    try:
        train_data = read_jsonl_basic('../util/news_multilabel_train_data.jsonl')
        val_data = read_jsonl_basic('../util/news_multilabel_val_data.jsonl')
        test_data = read_jsonl_basic('../util/news_multilabel_test_data.jsonl')
    except FileNotFoundError as e:
        print(f"❌ Ошибка загрузки файла: {e}")
        return

    print(f"📊 Данные: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Определяем количество меток из данных
    if train_data:
        n_labels = len(train_data[0]['binary_labels'])
        print(f"📊 Количество меток: {n_labels}")
    else:
        print("❌ Нет тренировочных данных!")
        return

    # 1. Анализ распределения меток
    print("\n" + "=" * 50)
    classifier = MultiLabelCatBoostClassifier(n_labels=n_labels, iterations=300, verbose=100)
    classifier.analyze_label_distribution(train_data)

    # 2. Сравнение гиперпараметров (опционально)
    print("\n" + "=" * 50)
    print("🔬 Сравнение гиперпараметров CatBoost (опциональный шаг)...")
    try:
        classifier.compare_hyperparameters(
            train_data[:1000],  # Используем подвыборку для быстрого тестирования
            val_data[:200] if val_data else None,
            iterations_list=[100, 200],
            depth_list=[4, 6],
            learning_rate_list=[0.05, 0.1]
        )
    except Exception as e:
        print(f"⚠️  Ошибка при сравнении гиперпараметров: {e}")
        print("Продолжаем с iterations=300, depth=6, learning_rate=0.1")

    # 3. Обучаем финальную модель
    print("\n" + "=" * 50)
    print("🐱 ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ CATBOOST...")
    classifier = MultiLabelCatBoostClassifier(
        n_labels=n_labels,
        iterations=300,
        depth=6,
        learning_rate=0.1,
        verbose=100
    )
    classifier.train(train_data, val_data)

    # 5. Оценка на тестовых данных
    print("\n" + "=" * 50)
    print("📊 ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ:")
    try:
        accuracy, hamming = classifier.evaluate(test_data)
        print(f"   Итоговая точность: {accuracy:.4f}")
        print(f"   Итоговая потеря Хэмминга: {hamming:.4f}")
    except Exception as e:
        print(f"❌ Ошибка при оценке модели: {e}")

    # 6. Сохраняем модель
    try:
        classifier.save_model("multilabel_catboost.pkl")
    except Exception as e:
        print(f"❌ Ошибка при сохранении модели: {e}")

    # 7. Строим графики
    print("\n" + "=" * 50)
    print("📈 ПОСТРОЕНИЕ ГРАФИКОВ:")
    try:
        classifier.plot_training_history()
        classifier.plot_feature_importances(top_n=20)
        classifier.plot_learning_curves(max_labels=3)
        classifier.plot_confusion_matrices(test_data, max_classes=4)
    except Exception as e:
        print(f"⚠️  Ошибка при построении графиков: {e}")


# Простой способ быстро обучить модель
def quick_train_catboost(train_file, val_file=None, n_labels=14, iterations=300):
    """
    Быстрое обучение CatBoost модели из файлов
    """

    # Загружаем данные
    def load_jsonl(filepath):
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file) if val_file else None

    # Обучаем модель
    classifier = MultiLabelCatBoostClassifier(
        n_labels=n_labels,
        iterations=iterations,
        verbose=0  # Без вывода логов
    )
    classifier.train(train_data, val_data)

    return classifier


if __name__ == "__main__":
    main()