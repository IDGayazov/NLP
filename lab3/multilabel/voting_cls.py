from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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


class SVMLinearClassifierWithProba(BaseEstimator, ClassifierMixin):
    """
    Обертка для LinearSVC с поддержкой вероятностей
    """

    def __init__(self, random_state=42, max_iter=1000, class_weight='balanced', dual=False, C=1.0):
        self.random_state = random_state
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.dual = dual
        self.C = C
        self.svm = LinearSVC(
            random_state=random_state,
            max_iter=max_iter,
            class_weight=class_weight,
            dual=dual,
            C=C
        )
        self.calibrator = None

    def get_params(self, deep=True):
        """Получение параметров для совместимости с scikit-learn"""
        params = {
            'random_state': self.random_state,
            'max_iter': self.max_iter,
            'class_weight': self.class_weight,
            'dual': self.dual,
            'C': self.C
        }
        if deep:
            # Рекурсивно получаем параметры вложенных объектов
            params['svm'] = self.svm
            if self.calibrator is not None:
                params['calibrator'] = self.calibrator
        return params

    def set_params(self, **params):
        """Установка параметров для совместимости с scikit-learn"""
        for key, value in params.items():
            if key == 'svm':
                self.svm = value
            elif key == 'calibrator':
                self.calibrator = value
            else:
                setattr(self, key, value)
        return self

    def fit(self, X, y):
        """Обучение SVM с калибрацией вероятностей"""
        self.svm.fit(X, y)
        # Калибрация для получения вероятностей
        unique_classes = np.unique(y)
        cv_value = min(3, len(unique_classes)) if len(unique_classes) > 1 else 2
        self.calibrator = CalibratedClassifierCV(self.svm, cv=cv_value, method='sigmoid')
        self.calibrator.fit(X, y)
        return self

    def predict(self, X):
        """Предсказание классов"""
        return self.svm.predict(X)

    def predict_proba(self, X):
        """Предсказание вероятностей через калибрацию"""
        if self.calibrator is None:
            # Если калибратор не обучен, возвращаем равномерные вероятности
            pred = self.svm.predict(X)
            proba = np.zeros((len(pred), 2))
            for i, p in enumerate(pred):
                proba[i, int(p)] = 1.0
            return proba
        return self.calibrator.predict_proba(X)

    def __sklearn_clone__(self):
        """Метод для правильного клонирования в sklearn"""
        import copy
        return copy.deepcopy(self)


class MultiLabelVotingClassifier:
    """
    Многометочный классификатор на основе Voting (голосования)
    """

    def __init__(self, n_labels=14, voting='soft', estimators=None,
                 weights=None, n_jobs=-1, random_state=42):
        """
        Args:
            n_labels: количество меток/классов
            voting: тип голосования ('hard' или 'soft')
            estimators: список базовых классификаторов
            weights: веса классификаторов
            n_jobs: количество ядер для параллельной обработки
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
        self.voting = voting
        self.estimators = estimators
        self.weights = weights
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.estimator_names = []  # названия классификаторов

    def _create_base_estimators(self):
        """Создает список базовых классификаторов"""
        if self.estimators is None:
            # По умолчанию используем 3 классификатора (без SVM из-за проблем с Voting)
            return [
                ('lr', LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight='balanced',
                    C=0.1
                )),
                ('dt', DecisionTreeClassifier(
                    random_state=self.random_state,
                    max_depth=10,
                    class_weight='balanced',
                    min_samples_split=10
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=50,
                    random_state=self.random_state,
                    class_weight='balanced',
                    max_depth=10,
                    n_jobs=self.n_jobs
                ))
            ]
        else:
            return self.estimators

    def _create_voting_classifier(self):
        """Создает Voting классификатор"""
        estimators = self._create_base_estimators()

        # Сохраняем названия классификаторов
        self.estimator_names = [name for name, _ in estimators]

        # Для soft voting проверяем, что все классификаторы имеют predict_proba
        if self.voting == 'soft':
            # Проверяем каждый классификатор
            valid_estimators = []
            for name, estimator in estimators:
                if hasattr(estimator, 'predict_proba') or hasattr(estimator, '_get_tags'):
                    valid_estimators.append((name, estimator))
                else:
                    print(
                        f"   ⚠️  Классификатор {name} ({estimator.__class__.__name__}) не поддерживает predict_proba, пропускаем")

            if not valid_estimators:
                raise ValueError("Нет классификаторов с поддержкой predict_proba для soft voting")

            estimators = valid_estimators

        return VotingClassifier(
            estimators=estimators,
            voting=self.voting,
            weights=self.weights,
            n_jobs=self.n_jobs,
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
        voting_type = "Жесткое (Hard)" if self.voting == 'hard' else "Мягкое (Soft)"
        print(f"🎯 ОБУЧЕНИЕ МНОГОМЕТОЧНОГО VOTING КЛАССИФИКАТОРА ({voting_type} Voting)...")
        print("=" * 60)

        # Показываем архитектуру голосования
        estimators = self._create_base_estimators()

        print("📊 Архитектура голосования:")
        print(f"   Тип голосования: {voting_type}")
        print(f"   Классификаторы ({len(estimators)}):")
        for name, estimator in estimators:
            print(f"     - {name}: {estimator.__class__.__name__}")

        if self.weights:
            print(f"   Веса классификаторов: {self.weights}")

        # Подготовка данных
        X_train, y_train = self.prepare_data(train_data)

        print(f"\n📊 Размер тренировочных данных: {len(X_train)}")
        print(f"📊 Количество меток: {self.n_labels}")
        print(f"📊 Формат меток: {y_train.shape}")

        # Инициализация и обучение векторизатора
        print("\n📊 Векторизация текстов...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        print(f"   Размерность признаков: {X_train_vec.shape}")

        # Обучаем отдельный классификатор для каждой метки
        print(f"\n🗳️ Обучение Voting для каждой метки ({voting_type})...")

        # Сброс списков
        self.estimators_ = []
        self.single_class_labels = set()

        for label_idx in range(self.n_labels):
            y_single = y_train[:, label_idx]
            unique_classes = np.unique(y_single)

            if len(unique_classes) < 2:
                # Если только один класс, используем DummyClassifier
                print(f"   Метка {label_idx}: только один класс ({unique_classes[0]}) - используем DummyClassifier")
                clf = DummyClassifier(strategy='constant', constant=unique_classes[0])
                clf.fit(X_train_vec, y_single)
                self.single_class_labels.add(label_idx)
            else:
                # Если два класса, используем Voting
                print(f"   Метка {label_idx}: обучение Voting...")
                try:
                    clf = self._create_voting_classifier()
                    clf.fit(X_train_vec, y_single)
                    print(f"      Успешно обучено!")
                except Exception as e:
                    print(f"   ⚠️ Ошибка при обучении Voting для метки {label_idx}: {e}")
                    print(f"   Используем LogisticRegression вместо Voting")
                    clf = LogisticRegression(
                        random_state=self.random_state,
                        max_iter=1000,
                        class_weight='balanced'
                    )
                    clf.fit(X_train_vec, y_single)

            self.estimators_.append(clf)

            # Прогресс
            if (label_idx + 1) % 3 == 0 or (label_idx + 1) == self.n_labels:
                print(f"   Прогресс: {label_idx + 1}/{self.n_labels} меток обучено")

        self.is_trained = True

        # Оценка на тренировочных данных
        y_pred_train = self._predict_from_estimators(X_train_vec)
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
                # Voting возвращает вероятности для soft voting
                try:
                    prob = clf.predict_proba(X_vec)
                    probabilities.append(prob[:, 1])  # вероятность класса 1
                except:
                    # Для hard voting или если predict_proba не доступен
                    pred = clf.predict(X_vec).astype(float)
                    probabilities.append(pred)  # используем предсказания как псевдовероятности
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
            voting_type = "Hard" if self.voting == 'hard' else "Soft"
            print(f"\n📊 ОЦЕНКА МОДЕЛИ VOTING ({voting_type}):")
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

    def _analyze_model(self):
        """
        Анализ обученной модели Voting
        """
        print(f"\n🗳️ АНАЛИЗ МОДЕЛИ VOTING ({self.voting.upper()}):")
        print("=" * 60)

        # Собираем статистику
        voting_models = 0
        dummy_models = 0
        lr_models = 0

        for idx, clf in enumerate(self.estimators_):
            if idx in self.single_class_labels:
                dummy_models += 1
            elif isinstance(clf, VotingClassifier):
                voting_models += 1
            else:
                lr_models += 1  # LogisticRegression как fallback

        print(f"   Всего меток: {self.n_labels}")
        print(f"   Voting моделей: {voting_models}")
        print(f"   Dummy моделей: {dummy_models}")
        print(f"   LogisticRegression моделей (fallback): {lr_models}")
        print(f"   Классификаторов в ансамбле: {len(self.estimator_names)}")
        print(f"   Классификаторы: {', '.join(self.estimator_names)}")
        print(f"   Тип голосования: {self.voting}")

        if self.weights:
            print(f"   Веса: {self.weights}")

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
            'voting': self.voting,
            'estimators_list': self.estimators,
            'weights': self.weights,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'estimator_names': self.estimator_names
        }, filename)
        print(f"💾 Модель Voting сохранена: {filename}")

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
        self.voting = loaded.get('voting', 'soft')
        self.estimators = loaded.get('estimators_list', None)
        self.weights = loaded.get('weights', None)
        self.n_jobs = loaded.get('n_jobs', -1)
        self.random_state = loaded.get('random_state', 42)
        self.estimator_names = loaded.get('estimator_names', [])
        self.is_trained = True
        print(f"📥 Модель Voting загружена: {filename}")

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

            sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',
                        xticklabels=['Pred 0', 'Pred 1'],
                        yticklabels=['True 0', 'True 1'],
                        ax=axes[i])
            axes[i].set_title(f'Confusion Matrix - Class {label_idx}')
            axes[i].set_ylabel('True Label')
            axes[i].set_xlabel('Predicted Label')

        voting_type = "Hard" if self.voting == 'hard' else "Soft"
        plt.suptitle(f'Матрицы ошибок (Voting {voting_type})', fontsize=14)
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
                print(f"   Класс {val}: {count} примеров ({percentage:.1f}%)")
            if len(unique) < 2:
                print(f"   ⚠️  Только один класс!")
            print()


# Пример использования
def main():
    """
    Пример использования многометочного Voting классификатора
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
    classifier = MultiLabelVotingClassifier(
        n_labels=n_labels,
        voting='soft',  # Можно изменить на 'hard' для Hard Voting
        n_jobs=-1
    )
    classifier.analyze_label_distribution(train_data)

    # 3. Обучаем финальную модель
    print("\n" + "=" * 50)
    print("🗳️ ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ VOTING...")

    # Выберите тип голосования:
    # voting_type = 'hard'  # Hard Voting (голосование по большинству)
    voting_type = 'soft'  # Soft Voting (усреднение вероятностей)

    # Для Voting используем только классификаторы, которые точно работают с VotingClassifier
    classifier = MultiLabelVotingClassifier(
        n_labels=n_labels,
        voting=voting_type,
        estimators=[
            ('lr', LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced',
                C=0.1
            )),
            ('dt', DecisionTreeClassifier(
                random_state=42,
                max_depth=8,
                class_weight='balanced',
                min_samples_split=10
            )),
            ('rf', RandomForestClassifier(
                n_estimators=50,
                random_state=42,
                class_weight='balanced',
                max_depth=10,
                n_jobs=-1
            ))
        ],
        # Можно задать веса для классификаторов (опционально)
        # weights=[2, 1, 1.5],
        n_jobs=-1,
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
        filename = "multilabel_voting_hard.pkl" if voting_type == 'hard' else "multilabel_voting_soft.pkl"
        classifier.save_model(filename)
    except Exception as e:
        print(f"❌ Ошибка при сохранении модели: {e}")

    # 7. Строим графики
    print("\n" + "=" * 50)
    print("📈 ПОСТРОЕНИЕ ГРАФИКОВ:")
    try:
        classifier.plot_training_history()
        classifier.plot_confusion_matrices(test_data, max_classes=4)
    except Exception as e:
        print(f"⚠️  Ошибка при построении графиков: {e}")

if __name__ == "__main__":
    main()