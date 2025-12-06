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


class MultiLabelRandomForestClassifier:
    """
    Многометочный классификатор на основе Случайного леса
    """

    def __init__(self, n_estimators=100, max_depth=None, n_labels=14, random_state=42):
        """
        Args:
            n_estimators: количество деревьев в лесу
            max_depth: максимальная глубина деревьев
            n_labels: количество меток/классов
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
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.feature_importances_ = None  # важность признаков

    def _create_random_forest(self, n_estimators, max_depth):
        """Создает Random Forest классификатор"""
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1,  # используем все ядра процессора
            min_samples_split=5,
            min_samples_leaf=2
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
        print("🎯 ОБУЧЕНИЕ МНОГОМЕТОЧНОГО СЛУЧАЙНОГО ЛЕСА...")

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
        print("🌲 Обучение случайного леса для каждой метки...")

        # Сброс списков
        self.estimators_ = []
        self.single_class_labels = set()
        self.feature_importances_ = np.zeros((self.n_labels, X_train_vec.shape[1]))

        for label_idx in range(self.n_labels):
            y_single = y_train[:, label_idx]
            unique_classes = np.unique(y_single)

            if len(unique_classes) < 2:
                # Если только один класс, используем DummyClassifier
                print(f"   Метка {label_idx}: только один класс ({unique_classes[0]}) - используем DummyClassifier")
                clf = DummyClassifier(strategy='constant', constant=unique_classes[0])
                clf.fit(X_train_vec, y_single)  # DummyClassifier тоже нужно обучить!
                self.single_class_labels.add(label_idx)
            else:
                # Если два класса, используем Random Forest
                print(f"   Метка {label_idx}: обучение Random Forest...")
                clf = self._create_random_forest(self.n_estimators, self.max_depth)
                clf.fit(X_train_vec, y_single)

                # Сохраняем важность признаков
                if hasattr(clf, 'feature_importances_'):
                    self.feature_importances_[label_idx] = clf.feature_importances_

            self.estimators_.append(clf)

        self.is_trained = True

        # Оценка на тренировочных данных
        y_pred_train = self._predict_from_estimators(X_train_vec)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_hamming = hamming_loss(y_train, y_pred_train)

        print(f"✅ Точность на train: {train_accuracy:.3f}")
        print(f"✅ Потеря Хэмминга на train: {train_hamming:.3f}")
        print(f"✅ Метки с одним классом: {sorted(self.single_class_labels)}")

        self.loss_history.append(train_hamming)
        self.accuracy_history.append(train_accuracy)

        # Оценка на валидации, если есть
        if val_data:
            val_accuracy, val_hamming = self.evaluate(val_data, verbose=False)
            print(f"✅ Точность на val: {val_accuracy:.3f}")
            print(f"✅ Потеря Хэмминга на val: {val_hamming:.3f}")
            self.val_loss_history.append(val_hamming)
            self.val_accuracy_history.append(val_accuracy)

        # Покажем важные признаки для меток с двумя классами
        self._show_important_features(top_n=10)

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
                # Random Forest возвращает вероятности напрямую
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
            print(f"\n📊 ОЦЕНКА МОДЕЛИ:")
            print(f"   Точность: {accuracy:.3f}")
            print(f"   Потеря Хэмминга: {hamming:.3f}")

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
                    print(f"      Accuracy: {np.mean(y_true == y_pred_single):.3f}")
                elif i in self.single_class_labels:
                    print(f"\n   Класс {i} (один класс в тренировочных данных):")
                    print(f"      Все предсказания: {np.unique(y_pred_single)[0]}")
                    print(f"      Accuracy: {np.mean(y_true == y_pred_single):.3f}")
                else:
                    print(f"\n   Класс {i}:")
                    try:
                        print(classification_report(y_true, y_pred_single,
                                                    target_names=[f'Отсутствует({i})', f'Присутствует({i})'],
                                                    zero_division=0))
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

        print(f"\n🌳 ТОП-{top_n} ВАЖНЫХ ПРИЗНАКОВ ДЛЯ МЕТОК С ДВУМЯ КЛАССАМИ:")

        for label_idx in range(self.n_labels):
            if label_idx not in self.single_class_labels and np.sum(self.feature_importances_[label_idx]) > 0:

                print(f"\n   КЛАСС {label_idx}:")

                # Получаем индексы самых важных признаков
                importance_scores = self.feature_importances_[label_idx]
                top_indices = np.argsort(importance_scores)[-top_n:][::-1]

                print(f"      Самые важные признаки:")
                for i, idx in enumerate(top_indices[:top_n]):
                    if idx < len(feature_names):
                        print(f"        {i + 1}. {feature_names[idx]}: {importance_scores[idx]:.4f}")

    def _analyze_model(self):
        """
        Анализ обученной модели Random Forest
        """
        print("\n🌲 АНАЛИЗ МОДЕЛИ RANDOM FOREST:")
        print("=" * 50)

        # Подсчитываем статистику по деревьям
        tree_depths = []
        leaf_counts = []

        for idx, clf in enumerate(self.estimators_):
            if idx not in self.single_class_labels:
                if hasattr(clf, 'estimators_'):
                    for tree in clf.estimators_:
                        tree_depths.append(tree.get_depth())
                        leaf_counts.append(tree.get_n_leaves())

        if tree_depths:
            print(f"   Средняя глубина деревьев: {np.mean(tree_depths):.1f}")
            print(f"   Максимальная глубина деревьев: {np.max(tree_depths)}")
            print(f"   Среднее количество листьев: {np.mean(leaf_counts):.1f}")
            print(f"   Общее количество деревьев: {len(tree_depths)}")
        else:
            print("   Нет деревьев для анализа (только DummyClassifiers)")

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
            'feature_importances': self.feature_importances_,
            'loss_history': self.loss_history,
            'val_loss_history': self.val_loss_history,
            'accuracy_history': self.accuracy_history,
            'val_accuracy_history': self.val_accuracy_history,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state
        }, filename)
        print(f"💾 Модель сохранена: {filename}")

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
        self.n_estimators = loaded.get('n_estimators', 100)
        self.max_depth = loaded.get('max_depth', None)
        self.random_state = loaded.get('random_state', 42)
        self.is_trained = True
        print(f"📥 Модель загружена: {filename}")

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

        ax.barh(y_pos, top_importances, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()  # Самые важные сверху
        ax.set_xlabel('Важность признака')
        ax.set_title(f'Топ-{top_n} самых важных признаков')

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

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
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
                                n_estimators_list=[50, 100, 200],
                                max_depth_list=[None, 10, 20, 30]):
        """
        Сравнение разных гиперпараметров Random Forest
        """
        print("🔬 СРАВНЕНИЕ РАЗНЫХ ГИПЕРПАРАМЕТРОВ RANDOM FOREST:")
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

        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                print(f"\n📊 Обучение с n_estimators={n_estimators}, max_depth={max_depth}:")

                # Обучаем отдельный классификатор для каждой метки
                estimators_temp = []
                for label_idx in range(self.n_labels):
                    y_single = y_train[:, label_idx]
                    unique_classes = np.unique(y_single)

                    if len(unique_classes) < 2:
                        clf = DummyClassifier(strategy='constant', constant=unique_classes[0])
                        clf.fit(X_train_vec, y_single)
                    else:
                        clf = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=self.random_state,
                            class_weight='balanced',
                            n_jobs=-1
                        )
                        clf.fit(X_train_vec, y_single)

                    estimators_temp.append(clf)

                # Функция для предсказания с заданными estimators
                def predict_with_estimators(X_vec, estimators):
                    predictions = []
                    for clf in estimators:
                        pred = clf.predict(X_vec)
                        predictions.append(pred)
                    return np.array(predictions).T

                # Оценка на тренировочных данных
                y_pred_train = predict_with_estimators(X_train_vec, estimators_temp)
                train_accuracy = accuracy_score(y_train, y_pred_train)
                train_hamming = hamming_loss(y_train, y_pred_train)

                # Оценка на валидации, если есть
                val_accuracy = None
                val_hamming = None
                if val_data:
                    y_pred_val = predict_with_estimators(X_val_vec, estimators_temp)
                    val_accuracy = accuracy_score(y_val, y_pred_val)
                    val_hamming = hamming_loss(y_val, y_pred_val)
                    print(f"   Train Accuracy: {train_accuracy:.3f}, Val Accuracy: {val_accuracy:.3f}")
                    print(f"   Train Hamming: {train_hamming:.3f}, Val Hamming: {val_hamming:.3f}")
                else:
                    print(f"   Train Accuracy: {train_accuracy:.3f}")
                    print(f"   Train Hamming: {train_hamming:.3f}")

                results.append({
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'train_accuracy': train_accuracy,
                    'train_hamming': train_hamming,
                    'val_accuracy': val_accuracy,
                    'val_hamming': val_hamming
                })

        # Построение графиков сравнения
        self._plot_hyperparameter_comparison(results)

    def _plot_hyperparameter_comparison(self, results):
        """Построение графиков сравнения разных гиперпараметров"""
        if not results:
            print("Нет результатов для построения графиков")
            return

        # Создаем сетку для визуализации
        n_estimators_list = sorted(list(set(r['n_estimators'] for r in results)))
        max_depth_list = sorted(list(set(r['max_depth'] for r in results)), key=lambda x: x if x is not None else 0)

        # Создаем матрицы для heatmap
        train_acc_matrix = np.zeros((len(max_depth_list), len(n_estimators_list)))
        val_acc_matrix = np.zeros((len(max_depth_list), len(n_estimators_list)))

        for r in results:
            i = max_depth_list.index(r['max_depth'])
            j = n_estimators_list.index(r['n_estimators'])
            train_acc_matrix[i, j] = r['train_accuracy']
            if r['val_accuracy'] is not None:
                val_acc_matrix[i, j] = r['val_accuracy']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Heatmap для тренировочной точности
        sns.heatmap(train_acc_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=n_estimators_list,
                    yticklabels=[str(d) if d is not None else 'None' for d in max_depth_list],
                    ax=axes[0])
        axes[0].set_xlabel('Количество деревьев')
        axes[0].set_ylabel('Максимальная глубина')
        axes[0].set_title('Train Accuracy')

        # Heatmap для валидационной точности (если есть)
        if np.sum(val_acc_matrix) > 0:
            sns.heatmap(val_acc_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                        xticklabels=n_estimators_list,
                        yticklabels=[str(d) if d is not None else 'None' for d in max_depth_list],
                        ax=axes[1])
            axes[1].set_xlabel('Количество деревьев')
            axes[1].set_ylabel('Максимальная глубина')
            axes[1].set_title('Validation Accuracy')
        else:
            axes[1].axis('off')

        plt.tight_layout()
        plt.show()


# Пример использования
def main():
    """
    Пример использования многометочного Random Forest классификатора
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
    classifier = MultiLabelRandomForestClassifier(n_labels=n_labels, n_estimators=100)
    classifier.analyze_label_distribution(train_data)

    # 2. Сравнение гиперпараметров (опционально)
    print("\n" + "=" * 50)
    print("🔬 Сравнение гиперпараметров Random Forest (опциональный шаг)...")
    try:
        classifier.compare_hyperparameters(
            train_data,
            val_data,
            n_estimators_list=[50, 100, 150],
            max_depth_list=[None, 10, 20]
        )
    except Exception as e:
        print(f"⚠️  Ошибка при сравнении гиперпараметров: {e}")
        print("Продолжаем с n_estimators=100, max_depth=None")

    # 3. Обучаем финальную модель
    print("\n" + "=" * 50)
    print("🌲 ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ RANDOM FOREST...")
    classifier = MultiLabelRandomForestClassifier(
        n_labels=n_labels,
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    classifier.train(train_data, val_data)

    # 5. Оценка на тестовых данных
    print("\n" + "=" * 50)
    print("📊 ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ:")
    try:
        accuracy, hamming = classifier.evaluate(test_data)
        print(f"   Итоговая точность: {accuracy:.3f}")
        print(f"   Итоговая потеря Хэмминга: {hamming:.3f}")
    except Exception as e:
        print(f"❌ Ошибка при оценке модели: {e}")

    # 6. Сохраняем модель
    try:
        classifier.save_model("multilabel_random_forest.pkl")
    except Exception as e:
        print(f"❌ Ошибка при сохранении модели: {e}")

    # 7. Строим графики
    print("\n" + "=" * 50)
    print("📈 ПОСТРОЕНИЕ ГРАФИКОВ:")
    try:
        classifier.plot_training_history()
        classifier.plot_feature_importances(top_n=20)
        classifier.plot_confusion_matrices(test_data, max_classes=4)
    except Exception as e:
        print(f"⚠️  Ошибка при построении графиков: {e}")


# Простой способ быстро обучить модель
def quick_train_random_forest(train_file, val_file=None, n_labels=14, n_estimators=100):
    """
    Быстрое обучение Random Forest модели из файлов
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
    classifier = MultiLabelRandomForestClassifier(
        n_labels=n_labels,
        n_estimators=n_estimators
    )
    classifier.train(train_data, val_data)

    return classifier


if __name__ == "__main__":
    main()