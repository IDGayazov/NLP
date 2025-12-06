from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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


class MultiLabelBaggingClassifier:
    """
    Многометочный классификатор на основе Bagging (бэггинга)
    """

    def __init__(self, n_labels=14, base_estimator='logistic', n_estimators=10,
                 max_samples=1.0, max_features=1.0, random_state=42):
        """
        Args:
            n_labels: количество меток/классов
            base_estimator: базовый классификатор ('logistic', 'tree' или объект классификатора)
            n_estimators: количество базовых классификаторов
            max_samples: доля примеров для обучения каждого классификатора
            max_features: доля признаков для обучения каждого классификатора
            random_state: seed для воспроизводимости
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
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.oob_scores_ = []  # out-of-bag оценки

    def _create_base_estimator(self):
        """Создает базовый классификатор"""
        if isinstance(self.base_estimator, str):
            if self.base_estimator == 'logistic':
                return LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight='balanced'
                )
            elif self.base_estimator == 'tree':
                return DecisionTreeClassifier(
                    random_state=self.random_state,
                    max_depth=10,
                    class_weight='balanced'
                )
            else:
                raise ValueError(f"Неизвестный base_estimator: {self.base_estimator}")
        else:
            return self.base_estimator

    def _create_bagging_classifier(self):
        """Создает Bagging классификатор"""
        base_estimator = self._create_base_estimator()

        return BaggingClassifier(
            estimator=base_estimator,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=True,  # использовать bootstrap выборку
            bootstrap_features=False,  # не использовать bootstrap для признаков
            oob_score=True,  # вычислять out-of-bag score
            random_state=self.random_state,
            n_jobs=-1,  # использовать все ядра процессора
            verbose=0
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
        print("🎯 ОБУЧЕНИЕ МНОГОМЕТОЧНОГО BAGGING КЛАССИФИКАТОРА...")

        base_name = "Логистическая регрессия" if self.base_estimator == 'logistic' else "Дерево решений"
        print(f"📊 Базовый классификатор: {base_name}")
        print(f"📊 Количество классификаторов: {self.n_estimators}")

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
        print(f"   Размерность признаков: {X_train_vec.shape}")

        # Обучаем отдельный классификатор для каждой метки
        print("👜 Обучение Bagging для каждой метки...")

        # Сброс списков
        self.estimators_ = []
        self.single_class_labels = set()
        self.oob_scores_ = []

        for label_idx in range(self.n_labels):
            y_single = y_train[:, label_idx]
            unique_classes = np.unique(y_single)

            if len(unique_classes) < 2:
                # Если только один класс, используем DummyClassifier
                print(f"   Метка {label_idx}: только один класс ({unique_classes[0]}) - используем DummyClassifier")
                clf = DummyClassifier(strategy='constant', constant=unique_classes[0])
                clf.fit(X_train_vec, y_single)
                self.single_class_labels.add(label_idx)
                self.oob_scores_.append(None)
            else:
                # Если два класса, используем Bagging
                print(f"   Метка {label_idx}: обучение Bagging...")
                clf = self._create_bagging_classifier()
                clf.fit(X_train_vec, y_single)

                # Сохраняем out-of-bag score
                if hasattr(clf, 'oob_score_'):
                    self.oob_scores_.append(clf.oob_score_)
                    print(f"      Out-of-bag score: {clf.oob_score_:.4f}")
                else:
                    self.oob_scores_.append(None)

            self.estimators_.append(clf)

            # Прогресс
            if (label_idx + 1) % 5 == 0 or (label_idx + 1) == self.n_labels:
                print(f"   Прогресс: {label_idx + 1}/{self.n_labels} меток обучено")

        self.is_trained = True

        # Оценка на тренировочных данных
        y_pred_train = self._predict_from_estimators(X_train_vec)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_hamming = hamming_loss(y_train, y_pred_train)

        print(f"\n✅ Точность на train: {train_accuracy:.4f}")
        print(f"✅ Потеря Хэмминга на train: {train_hamming:.4f}")
        print(f"✅ Метки с одним классом: {sorted(self.single_class_labels)}")

        # Средний out-of-bag score
        valid_oob_scores = [s for s in self.oob_scores_ if s is not None]
        if valid_oob_scores:
            print(f"✅ Средний out-of-bag score: {np.mean(valid_oob_scores):.4f}")

        self.loss_history.append(train_hamming)
        self.accuracy_history.append(train_accuracy)

        # Оценка на валидации, если есть
        if val_data:
            val_accuracy, val_hamming = self.evaluate(val_data, verbose=False)
            print(f"✅ Точность на val: {val_accuracy:.4f}")
            print(f"✅ Потеря Хэмминга на val: {val_hamming:.4f}")
            self.val_loss_history.append(val_hamming)
            self.val_accuracy_history.append(val_accuracy)

        # Анализ обученной модели
        self._analyze_model()

    def _predict_from_estimators(self, X_vec):
        """Предсказание с использованием всех обученных классификаторов"""
        predictions = []
        for clf in self.estimators_:
            pred = clf.predict(X_vec)
            predictions.append(pred)
        return np.array(predictions).T

    def _predict_proba_from_estimators(self, X_vec):
        """Предсказание вероятностей с использованием всех обученных классификаторов"""
        probabilities = []
        for idx, clf in enumerate(self.estimators_):
            if idx in self.single_class_labels:
                # Для DummyClassifier просто возвращаем [1, 0] или [0, 1]
                pred = clf.predict(X_vec)
                prob = np.zeros((len(pred), 2))
                for i in range(len(pred)):
                    if pred[i] == 1:
                        prob[i] = [0, 1]  # [P(0), P(1)]
                    else:
                        prob[i] = [1, 0]  # [P(0), P(1)]
                probabilities.append(prob[:, 1])
            else:
                # Bagging возвращает вероятности через усреднение базовых классификаторов
                prob = clf.predict_proba(X_vec)
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
        predictions = self._predict_from_estimators(X_vec)
        probabilities = self._predict_proba_from_estimators(X_vec)

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

        y_pred = self._predict_from_estimators(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        hamming = hamming_loss(y_test, y_pred)

        if verbose:
            print(f"\n📊 ОЦЕНКА МОДЕЛИ BAGGING:")
            print(f"   Точность: {accuracy:.4f}")
            print(f"   Потеря Хэмминга: {hamming:.4f}")

            # Анализ меток с одним классом
            if self.single_class_labels:
                print(f"\n⚠️  Метки с одним классом в тренировочных данных: {sorted(self.single_class_labels)}")
                print("   (для этих меток использовался DummyClassifier)")

            # Out-of-bag scores
            valid_oob_scores = [s for s in self.oob_scores_ if s is not None]
            if valid_oob_scores:
                print(f"   Средний OOB score: {np.mean(valid_oob_scores):.4f}")

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
                    oob_score = f", OOB: {self.oob_scores_[i]:.4f}" if self.oob_scores_[i] is not None else ""
                    print(
                        f"      Out-of-bag score: {self.oob_scores_[i]:.4f}" if self.oob_scores_[i] is not None else "")
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

    def _analyze_model(self):
        """
        Анализ обученной модели Bagging
        """
        print("\n👜 АНАЛИЗ МОДЕЛИ BAGGING:")
        print("=" * 50)

        # Собираем статистику
        bagging_models = 0
        dummy_models = 0

        for idx, clf in enumerate(self.estimators_):
            if idx in self.single_class_labels:
                dummy_models += 1
            else:
                bagging_models += 1

        print(f"   Всего меток: {self.n_labels}")
        print(f"   Bagging моделей: {bagging_models}")
        print(f"   Dummy моделей: {dummy_models}")
        print(f"   Классификаторов в ансамбле: {self.n_estimators}")
        print(f"   Base estimator: {self.base_estimator}")

        # Out-of-bag scores статистика
        valid_oob_scores = [s for s in self.oob_scores_ if s is not None]
        if valid_oob_scores:
            print(f"\n   Статистика out-of-bag scores:")
            print(f"      Средний: {np.mean(valid_oob_scores):.4f}")
            print(f"      Минимальный: {np.min(valid_oob_scores):.4f}")
            print(f"      Максимальный: {np.max(valid_oob_scores):.4f}")
            print(f"      Стандартное отклонение: {np.std(valid_oob_scores):.4f}")

    def save_model(self, filename):
        """
        Сохранение модели
        """
        if not self.is_trained:
            print("⚠️  Модель не обучена. Нечего сохранять.")
            return

        joblib.dump({
            'estimators': self.estimators_,
            'vectorizer': self.vectorizer,
            'n_labels': self.n_labels,
            'single_class_labels': self.single_class_labels,
            'loss_history': self.loss_history,
            'val_loss_history': self.val_loss_history,
            'accuracy_history': self.accuracy_history,
            'val_accuracy_history': self.val_accuracy_history,
            'base_estimator': self.base_estimator,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'oob_scores': self.oob_scores_
        }, filename)
        print(f"💾 Модель Bagging сохранена: {filename}")

    def load_model(self, filename):
        """
        Загрузка модели
        """
        loaded = joblib.load(filename)
        self.estimators_ = loaded['estimators']
        self.vectorizer = loaded['vectorizer']
        self.n_labels = loaded.get('n_labels', 14)
        self.single_class_labels = loaded.get('single_class_labels', set())
        self.loss_history = loaded.get('loss_history', [])
        self.val_loss_history = loaded.get('val_loss_history', [])
        self.accuracy_history = loaded.get('accuracy_history', [])
        self.val_accuracy_history = loaded.get('val_accuracy_history', [])
        self.base_estimator = loaded.get('base_estimator', 'logistic')
        self.n_estimators = loaded.get('n_estimators', 10)
        self.max_samples = loaded.get('max_samples', 1.0)
        self.max_features = loaded.get('max_features', 1.0)
        self.random_state = loaded.get('random_state', 42)
        self.oob_scores_ = loaded.get('oob_scores', [])
        self.is_trained = True
        print(f"📥 Модель Bagging загружена: {filename}")

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

    def plot_oob_scores(self):
        """
        Визуализация out-of-bag scores
        """
        valid_oob_scores = [s for s in self.oob_scores_ if s is not None]
        valid_indices = [i for i, s in enumerate(self.oob_scores_) if s is not None]

        if not valid_oob_scores:
            print("Нет out-of-bag scores для визуализации")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # График out-of-bag scores по меткам
        axes[0].bar(valid_indices, valid_oob_scores, color='skyblue', alpha=0.7)
        axes[0].axhline(y=np.mean(valid_oob_scores), color='red', linestyle='--',
                        label=f'Среднее: {np.mean(valid_oob_scores):.4f}')
        axes[0].set_xlabel('Метка')
        axes[0].set_ylabel('Out-of-bag Score')
        axes[0].set_title('Out-of-bag Scores по меткам')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Гистограмма распределения out-of-bag scores
        axes[1].hist(valid_oob_scores, bins=15, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1].axvline(x=np.mean(valid_oob_scores), color='red', linestyle='--',
                        label=f'Среднее: {np.mean(valid_oob_scores):.4f}')
        axes[1].set_xlabel('Out-of-bag Score')
        axes[1].set_ylabel('Частота')
        axes[1].set_title('Распределение Out-of-bag Scores')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'Bagging с {self.n_estimators} классификаторами', fontsize=14)
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
        y_pred = self._predict_from_estimators(X_test_vec)

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

            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
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

    def compare_bagging_variants(self, train_data, val_data=None,
                                 n_estimators_list=[5, 10, 20, 50],
                                 base_estimators=['logistic', 'tree']):
        """
        Сравнение разных вариантов Bagging
        """
        print("🔬 СРАВНЕНИЕ РАЗНЫХ ВАРИАНТОВ BAGGING:")
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

        if val_data:
            X_val, y_val = self.prepare_data(val_data)
            X_val_vec = vectorizer_temp.transform(X_val)

        results = []

        for base_est in base_estimators:
            for n_est in n_estimators_list:
                print(f"\n📊 Base: {base_est}, n_estimators: {n_est}")

                # Обучаем только на первой метке с двумя классами для быстрого тестирования
                test_label = None
                for i in range(self.n_labels):
                    if len(np.unique(y_train[:, i])) >= 2:
                        test_label = i
                        break

                if test_label is None:
                    print("   Нет меток с двумя классами для тестирования")
                    continue

                y_single = y_train[:, test_label]

                # Создаем и обучаем Bagging классификатор
                if base_est == 'logistic':
                    base_clf = LogisticRegression(random_state=42, max_iter=1000)
                else:
                    base_clf = DecisionTreeClassifier(random_state=42, max_depth=10)

                clf = BaggingClassifier(
                    estimator=base_clf,
                    n_estimators=n_est,
                    random_state=42,
                    n_jobs=-1,
                    oob_score=True
                )

                clf.fit(X_train_vec, y_single)

                # Оценка на тренировочных данных
                y_pred_train = clf.predict(X_train_vec)
                train_accuracy = accuracy_score(y_single, y_pred_train)

                # Оценка на валидации, если есть
                val_accuracy = None
                if val_data:
                    y_val_single = y_val[:, test_label]
                    y_pred_val = clf.predict(X_val_vec)
                    val_accuracy = accuracy_score(y_val_single, y_pred_val)

                # Out-of-bag score
                oob_score = clf.oob_score_ if hasattr(clf, 'oob_score_') else None

                if val_data:
                    print(
                        f"   Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}, OOB: {oob_score:.4f}")
                else:
                    print(f"   Train Accuracy: {train_accuracy:.4f}, OOB: {oob_score:.4f}")

                results.append({
                    'base_estimator': base_est,
                    'n_estimators': n_est,
                    'train_accuracy': train_accuracy,
                    'val_accuracy': val_accuracy,
                    'oob_score': oob_score
                })

        # Визуализация результатов
        self._plot_bagging_comparison(results)

    def _plot_bagging_comparison(self, results):
        """Построение графиков сравнения разных вариантов Bagging"""
        if not results:
            print("Нет результатов для построения графиков")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Разделяем результаты по типам базовых классификаторов
        logistic_results = [r for r in results if r['base_estimator'] == 'logistic']
        tree_results = [r for r in results if r['base_estimator'] == 'tree']

        # График для логистической регрессии
        if logistic_results:
            n_estimators = [r['n_estimators'] for r in logistic_results]
            train_acc = [r['train_accuracy'] for r in logistic_results]

            axes[0].plot(n_estimators, train_acc, 'bo-', label='Train', linewidth=2, markersize=8)

            if logistic_results[0]['val_accuracy'] is not None:
                val_acc = [r['val_accuracy'] for r in logistic_results]
                axes[0].plot(n_estimators, val_acc, 'rs-', label='Validation', linewidth=2, markersize=8)

            if logistic_results[0]['oob_score'] is not None:
                oob_scores = [r['oob_score'] for r in logistic_results]
                axes[0].plot(n_estimators, oob_scores, 'g^-', label='OOB', linewidth=2, markersize=8)

            axes[0].set_xlabel('Количество классификаторов')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title('Bagging с логистической регрессией')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # График для дерева решений
        if tree_results:
            n_estimators = [r['n_estimators'] for r in tree_results]
            train_acc = [r['train_accuracy'] for r in tree_results]

            axes[1].plot(n_estimators, train_acc, 'bo-', label='Train', linewidth=2, markersize=8)

            if tree_results[0]['val_accuracy'] is not None:
                val_acc = [r['val_accuracy'] for r in tree_results]
                axes[1].plot(n_estimators, val_acc, 'rs-', label='Validation', linewidth=2, markersize=8)

            if tree_results[0]['oob_score'] is not None:
                oob_scores = [r['oob_score'] for r in tree_results]
                axes[1].plot(n_estimators, oob_scores, 'g^-', label='OOB', linewidth=2, markersize=8)

            axes[1].set_xlabel('Количество классификаторов')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Bagging с деревом решений')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.suptitle('Влияние количества классификаторов на точность Bagging', fontsize=14)
        plt.tight_layout()
        plt.show()


def main():
    """
    Пример использования многометочного Bagging классификатора
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
    classifier = MultiLabelBaggingClassifier(
        n_labels=n_labels,
        base_estimator='logistic',
        n_estimators=20,
        random_state=42
    )
    classifier.analyze_label_distribution(train_data)

    # 2. Сравнение вариантов Bagging (опционально)
    print("\n" + "=" * 50)
    print("🔬 Сравнение вариантов Bagging (опциональный шаг)...")
    try:
        # Используем подвыборку для быстрого тестирования
        classifier.compare_bagging_variants(
            train_data[:2000],
            val_data[:500] if val_data else None,
            n_estimators_list=[5, 10, 20],
            base_estimators=['logistic', 'tree']
        )
    except Exception as e:
        print(f"⚠️  Ошибка при сравнении вариантов: {e}")
        print("Продолжаем с Bagging на логистической регрессии, n_estimators=20")

    # 3. Обучаем финальную модель
    print("\n" + "=" * 50)
    print("👜 ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ BAGGING...")

    # Можно выбрать один из вариантов:
    # 1. Bagging на логистической регрессии
    # classifier = MultiLabelBaggingClassifier(
    #     n_labels=n_labels,
    #     base_estimator='logistic',
    #     n_estimators=20,
    #     random_state=42
    # )

    # 2. Bagging на дереве решений
    classifier = MultiLabelBaggingClassifier(
        n_labels=n_labels,
        base_estimator='tree',
        n_estimators=20,
        random_state=42
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
        classifier.save_model("multilabel_bagging.pkl")
    except Exception as e:
        print(f"❌ Ошибка при сохранении модели: {e}")

    # 7. Строим графики
    print("\n" + "=" * 50)
    print("📈 ПОСТРОЕНИЕ ГРАФИКОВ:")
    try:
        classifier.plot_training_history()
        classifier.plot_oob_scores()
        classifier.plot_confusion_matrices(test_data, max_classes=4)
    except Exception as e:
        print(f"⚠️  Ошибка при построении графиков: {e}")


# Простой способ быстро обучить модель
def quick_train_bagging(train_file, val_file=None, n_labels=14,
                        base_estimator='logistic', n_estimators=20):
    """
    Быстрое обучение Bagging модели из файлов
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
    classifier = MultiLabelBaggingClassifier(
        n_labels=n_labels,
        base_estimator=base_estimator,
        n_estimators=n_estimators
    )
    classifier.train(train_data, val_data)

    return classifier


if __name__ == "__main__":
    main()