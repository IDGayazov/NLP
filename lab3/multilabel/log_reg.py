import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, multilabel_confusion_matrix, hamming_loss

warnings.filterwarnings('ignore')


class MultiLabelClassifier:
    """
    Многометочный классификатор на основе логистической регрессии
    """

    def __init__(self, regularization='l2', C=1.0, n_labels=14):
        """
        Args:
            regularization: 'l1' или 'l2' регуляризация
            C: параметр регуляризации (меньше = сильнее регуляризация)
            n_labels: количество меток/классов
        """
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )

        self.n_labels = n_labels
        self.is_trained = False
        self.estimators_ = []  # будем хранить классификаторы для каждой метки отдельно
        self.single_class_labels = set()  # метки с только одним классом
        self.loss_history = []  # история потерь
        self.val_loss_history = []  # история потерь на валидации
        self.accuracy_history = []  # история точности
        self.val_accuracy_history = []  # история точности на валидации

    def _create_classifier(self, regularization, C):
        """Создает классификатор с учетом типа регуляризации"""
        if regularization == 'l1':
            return LogisticRegression(
                penalty='l1',
                C=C,
                random_state=42,
                solver='liblinear',
                max_iter=1000,
                class_weight='balanced'
            )
        else:
            return LogisticRegression(
                penalty='l2',
                C=C,
                random_state=42,
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced'
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
        print("🎯 ОБУЧЕНИЕ МНОГОМЕТОЧНОЙ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ...")

        # Подготовка данных
        X_train, y_train = self.prepare_data(train_data)

        print(f"📊 Размер тренировочных данных: {len(X_train)}")
        print(f"📊 Количество меток: {self.n_labels}")
        print(f"📊 Формат меток: {y_train.shape}")

        # Векторизация текстов
        print("📊 Векторизация текстов...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        print(f"   Размерность признаков: {X_train_vec.shape}")

        # Обучаем отдельный классификатор для каждой метки
        print("🤖 Обучение модели для каждой метки...")

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
                self.single_class_labels.add(label_idx)
            else:
                # Если два класса, используем LogisticRegression
                clf = self._create_classifier('l2', 1.0)

            clf.fit(X_train_vec, y_single)
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
        self._show_important_features(X_train_vec, top_n=5)

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
            else:
                prob = clf.predict_proba(X_vec)
            probabilities.append(prob[:, 1])  # вероятность класса 1
        return np.array(probabilities).T

    def predict(self, texts):
        """
        Предсказание для списка текстов
        """
        if not self.is_trained:
            raise Exception("Модель не обучена!")

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
                    print(classification_report(y_true, y_pred_single,
                                                target_names=[f'Отсутствует({i})', f'Присутствует({i})'],
                                                zero_division=0))

            # Матрицы ошибок для классов с двумя классами в тестовых данных
            print("\n📈 МАТРИЦЫ ОШИБОК ПО КЛАССАМ (только для меток с двумя классами в тестовых данных):")
            valid_labels = []
            for i in range(self.n_labels):
                if len(np.unique(y_test[:, i])) >= 2 and i not in self.single_class_labels:
                    valid_labels.append(i)

            if valid_labels:
                y_test_valid = y_test[:, valid_labels]
                y_pred_valid = y_pred[:, valid_labels]
                cm = multilabel_confusion_matrix(y_test_valid, y_pred_valid)

                for idx, label_idx in enumerate(valid_labels[:5]):  # Показываем только первые 5
                    print(f"\n   Класс {label_idx}:")
                    print(f"               Предсказано 0  Предсказано 1")
                    print(f"   Реально 0:     {cm[idx][0][0]:^10}        {cm[idx][0][1]:^10}")
                    print(f"   Реально 1:     {cm[idx][1][0]:^10}        {cm[idx][1][1]:^10}")
            else:
                print("   Нет меток с двумя классами в тестовых данных для построения матриц ошибок")

        return accuracy, hamming

    def _show_important_features(self, X_vec, top_n=5):
        """
        Показывает самые важные признаки для меток с двумя классами
        """
        feature_names = self.vectorizer.get_feature_names_out()

        print(f"\n🔍 ТОП-{top_n} ВАЖНЫХ ПРИЗНАКОВ ДЛЯ МЕТОК С ДВУМЯ КЛАССАМИ:")

        for idx, clf in enumerate(self.estimators_):
            if idx not in self.single_class_labels and hasattr(clf, 'coef_'):
                coef = clf.coef_[0] if len(clf.coef_.shape) > 1 else clf.coef_

                print(f"\n   КЛАСС {idx}:")

                # Положительные признаки (указывают на присутствие метки)
                print(f"      Признаки для класса 1:")
                pos_indices = np.argsort(coef)[-top_n:][::-1]
                for pos_idx in pos_indices:
                    if pos_idx < len(feature_names):
                        print(f"        {feature_names[pos_idx]}: {coef[pos_idx]:.3f}")

    def save_model(self, filename):
        """
        Сохранение модели
        """
        joblib.dump({
            'estimators': self.estimators_,
            'vectorizer': self.vectorizer,
            'n_labels': self.n_labels,
            'single_class_labels': self.single_class_labels,
            'loss_history': self.loss_history,
            'val_loss_history': self.val_loss_history,
            'accuracy_history': self.accuracy_history,
            'val_accuracy_history': self.val_accuracy_history
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
        self.loss_history = loaded.get('loss_history', [])
        self.val_loss_history = loaded.get('val_loss_history', [])
        self.accuracy_history = loaded.get('accuracy_history', [])
        self.val_accuracy_history = loaded.get('val_accuracy_history', [])
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

    def plot_confusion_matrices(self, test_data, max_classes=4):
        """
        Визуализация матриц ошибок для первых N классов с двумя классами
        """
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


# Пример использования
def main():
    """
    Пример использования многометочного классификатора
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
    classifier = MultiLabelClassifier(n_labels=n_labels)
    classifier.analyze_label_distribution(train_data)

    # 2. Обучаем модель
    print("\n" + "=" * 50)
    classifier.train(train_data, val_data)

    # 4. Оценка на тестовых данных
    print("\n" + "=" * 50)
    print("📊 ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ:")
    try:
        accuracy, hamming = classifier.evaluate(test_data)
        print(f"   Итоговая точность: {accuracy:.3f}")
        print(f"   Итоговая потеря Хэмминга: {hamming:.3f}")
    except Exception as e:
        print(f"❌ Ошибка при оценке модели: {e}")

    # 5. Сохраняем модель
    classifier.save_model("multilabel_classifier.pkl")

    # 6. Строим графики
    print("\n" + "=" * 50)
    print("📈 ПОСТРОЕНИЕ ГРАФИКОВ:")
    classifier.plot_training_history()
    classifier.plot_confusion_matrices(test_data, max_classes=4)


# Простой способ быстро обучить модель
def quick_train(train_file, val_file=None, n_labels=14):
    """
    Быстрое обучение модели из файлов
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
    classifier = MultiLabelClassifier(regularization='l2', C=1.0, n_labels=n_labels)
    classifier.train(train_data, val_data)

    return classifier


if __name__ == "__main__":
    main()