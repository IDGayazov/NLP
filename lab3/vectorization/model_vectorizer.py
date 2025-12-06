import logging
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from gensim.models import Word2Vec, FastText

from util.load_tokenize_dataset import load_tokenize_ds


class EmbeddingModels:
    """
    Класс для обучения и управления моделями эмбеддингов
    """

    def __init__(self, vector_size: int = 100, window: int = 5,
                 min_count: int = 2, workers: int = 4, epochs: int = 10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs

        # Модели
        self.word2vec_model = None
        self.fasttext_model = None
        self.glove_vectors = None

        # Метаданные обучения
        self.training_metadata = {}

        # Настройка логирования
        logging.basicConfig(
            format='%(asctime)s : %(levelname)s : %(message)s',
            level=logging.INFO
        )

    def train_word2vec(self, tokenized_texts: List[List[str]],
                       sg: int = 1, **kwargs) -> Word2Vec:
        """
        Обучение Word2Vec модели
        """
        print("🎯 Обучение Word2Vec модели...")

        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=sg,
            epochs=self.epochs,
            **kwargs
        )

        # Сохраняем метаданные
        self.training_metadata['word2vec'] = {
            'vocab_size': len(self.word2vec_model.wv),
            'training_date': datetime.now().isoformat(),
            'parameters': {
                'vector_size': self.vector_size,
                'window': self.window,
                'min_count': self.min_count,
                'sg': sg,
                'epochs': self.epochs
            }
        }

        print(f"✅ Word2Vec обучена. Размер словаря: {len(self.word2vec_model.wv)}")
        return self.word2vec_model

    def train_fasttext(self, tokenized_texts: List[List[str]],
                       sg: int = 1, **kwargs) -> FastText:
        """
        Обучение FastText модели
        """
        print("🎯 Обучение FastText модели...")

        self.fasttext_model = FastText(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=sg,
            epochs=self.epochs,
            **kwargs
        )

        # Сохраняем метаданные
        self.training_metadata['fasttext'] = {
            'vocab_size': len(self.fasttext_model.wv),
            'training_date': datetime.now().isoformat(),
            'parameters': {
                'vector_size': self.vector_size,
                'window': self.window,
                'min_count': self.min_count,
                'sg': sg,
                'epochs': self.epochs
            }
        }

        print(f"✅ FastText обучена. Размер словаря: {len(self.fasttext_model.wv)}")
        return self.fasttext_model

    def train_glove(self, tokenized_texts: List[List[str]],
                    corpus_file: str = "corpus.txt", **kwargs):
        """
        Обучение GloVe модели (через glove-python)
        """
        try:
            from glove import Corpus, Glove
        except ImportError:
            print("❌ glove-python не установлен. Установите: pip install glove-python")
            return None

        print("🎯 Обучение GloVe модели...")

        # Создаем корпус
        corpus = Corpus()
        corpus.fit(tokenized_texts, window=self.window)

        # Обучаем GloVe
        self.glove_model = Glove(no_components=self.vector_size,
                                 learning_rate=0.05)
        self.glove_model.fit(corpus.matrix, epochs=self.epochs,
                             no_threads=self.workers, verbose=True)
        self.glove_model.add_dictionary(corpus.dictionary)

        # Сохраняем метаданные
        self.training_metadata['glove'] = {
            'vocab_size': len(corpus.dictionary),
            'training_date': datetime.now().isoformat(),
            'parameters': {
                'vector_size': self.vector_size,
                'window': self.window,
                'epochs': self.epochs
            }
        }

        print(f"✅ GloVe обучена. Размер словаря: {len(corpus.dictionary)}")
        return self.glove_model

    def save_models(self, base_path: str = "embeddings") -> Dict[str, str]:
        """
        Сохранение всех обученных моделей

        Returns:
            Словарь с путями к сохраненным файлам
        """
        Path(base_path).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = {}

        # Сохраняем Word2Vec
        if self.word2vec_model:
            w2v_path = f"{base_path}/word2vec_{timestamp}.model"
            self.word2vec_model.save(w2v_path)

            # Также сохраняем векторы в формате word2vec
            w2v_vectors_path = f"{base_path}/word2vec_vectors_{timestamp}.kv"
            self.word2vec_model.wv.save(w2v_vectors_path)

            saved_paths['word2vec'] = w2v_path
            saved_paths['word2vec_vectors'] = w2v_vectors_path
            print(f"💾 Word2Vec сохранена: {w2v_path}")

        # Сохраняем FastText
        if self.fasttext_model:
            ft_path = f"{base_path}/fasttext_{timestamp}.model"
            self.fasttext_model.save(ft_path)

            ft_vectors_path = f"{base_path}/fasttext_vectors_{timestamp}.kv"
            self.fasttext_model.wv.save(ft_vectors_path)

            saved_paths['fasttext'] = ft_path
            saved_paths['fasttext_vectors'] = ft_vectors_path
            print(f"💾 FastText сохранена: {ft_path}")

        # Сохраняем GloVe
        if hasattr(self, 'glove_model') and self.glove_model:
            glove_path = f"{base_path}/glove_{timestamp}.model"

            # Сохраняем модель и векторы
            with open(glove_path, 'wb') as f:
                pickle.dump(self.glove_model, f)

            # Сохраняем векторы в текстовом формате
            glove_vectors_path = f"{base_path}/glove_vectors_{timestamp}.txt"
            self._save_glove_vectors(glove_vectors_path)

            saved_paths['glove'] = glove_path
            saved_paths['glove_vectors'] = glove_vectors_path
            print(f"💾 GloVe сохранена: {glove_path}")

        # Сохраняем метаданные
        metadata_path = f"{base_path}/training_metadata_{timestamp}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_metadata, f, ensure_ascii=False, indent=2)

        saved_paths['metadata'] = metadata_path
        print(f"💾 Метаданные сохранены: {metadata_path}")

        return saved_paths

    def _save_glove_vectors(self, filepath: str):
        """Сохранение GloVe векторов в текстовом формате"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for word, idx in self.glove_model.dictionary.items():
                vector = self.glove_model.word_vectors[idx]
                vector_str = ' '.join([str(x) for x in vector])
                f.write(f"{word} {vector_str}\n")

    def load_models(self, base_path: str = "embeddings") -> bool:
        """
        Загрузка ранее сохраненных моделей

        Returns:
            True если загрузка успешна
        """
        try:
            # Ищем последние файлы по шаблону
            embedding_files = list(Path(base_path).glob("*_*.model"))

            for file_path in embedding_files:
                filename = file_path.stem

                if filename.startswith('word2vec'):
                    self.word2vec_model = Word2Vec.load(str(file_path))
                    print(f"📥 Word2Vec загружена: {file_path}")

                elif filename.startswith('fasttext'):
                    self.fasttext_model = FastText.load(str(file_path))
                    print(f"📥 FastText загружена: {file_path}")

                elif filename.startswith('glove'):
                    with open(file_path, 'rb') as f:
                        self.glove_model = pickle.load(f)
                    print(f"📥 GloVe загружена: {file_path}")

            # Загружаем метаданные
            metadata_files = list(Path(base_path).glob("training_metadata_*.json"))
            if metadata_files:
                latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
                with open(latest_metadata, 'r', encoding='utf-8') as f:
                    self.training_metadata = json.load(f)
                print(f"📥 Метаданные загружены: {latest_metadata}")

            return True

        except Exception as e:
            print(f"❌ Ошибка загрузки моделей: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о всех обученных моделях
        """
        info = {}

        if self.word2vec_model:
            info['word2vec'] = {
                'vocab_size': len(self.word2vec_model.wv),
                'vector_size': self.vector_size,
                'most_frequent': list(self.word2vec_model.wv.index_to_key[:10])
            }

        if self.fasttext_model:
            info['fasttext'] = {
                'vocab_size': len(self.fasttext_model.wv),
                'vector_size': self.vector_size,
                'most_frequent': list(self.fasttext_model.wv.index_to_key[:10])
            }

        if hasattr(self, 'glove_model') and self.glove_model:
            info['glove'] = {
                'vocab_size': len(self.glove_model.dictionary),
                'vector_size': self.vector_size,
                'most_frequent': list(self.glove_model.dictionary.keys())[:10]
            }

        return info

    def train_all_models(self, tokenized_texts: List[List[str]],
                         save_path: str = "embeddings") -> Dict[str, Any]:
        """
        Обучение всех моделей и автоматическое сохранение

        Returns:
            Словарь с путями к сохраненным моделям и информацией
        """
        print("🚀 Запуск обучения всех моделей эмбеддингов...")

        # Обучаем модели
        self.train_word2vec(tokenized_texts)
        self.train_fasttext(tokenized_texts)
        self.train_glove(tokenized_texts)

        # Сохраняем модели
        saved_paths = self.save_models(save_path)

        # Получаем информацию о моделях
        model_info = self.get_model_info()

        results = {
            'saved_paths': saved_paths,
            'model_info': model_info,
            'training_metadata': self.training_metadata
        }

        print("✅ Все модели обучены и сохранены!")
        return results


def main():
    # tokenized_texts = [
    #     ["машинное", "обучение", "искусственный", "интеллект"],
    #     ["нейронные", "сети", "глубокое", "обучение"],
    #     ["татарский", "язык", "лингвистика", "обработка", "текста"],
    #     ["машинное", "обучение", "алгоритм", "данные"],
    #     ["татарский", "культура", "традиция", "язык"]
    # ]

    tokenized_texts = load_tokenize_ds(size=600)

    # Инициализация и обучение
    embedding_trainer = EmbeddingModels(
        vector_size=100,
        window=5,
        min_count=1,
        epochs=10
    )

    # Обучение всех моделей
    results = embedding_trainer.train_all_models(tokenized_texts)

    # Вывод информации
    print("\n📊 ИНФОРМАЦИЯ О МОДЕЛЯХ:")
    for model_name, info in results['model_info'].items():
        print(f"\n{model_name.upper()}:")
        print(f"  Размер словаря: {info['vocab_size']}")
        print(f"  Размер вектора: {info['vector_size']}")
        print(f"  Частые слова: {info['most_frequent']}")

    print(f"\n💾 Пути сохранения:")
    for file_type, path in results['saved_paths'].items():
        print(f"  {file_type}: {path}")

    # Пример использования векторов
    if embedding_trainer.word2vec_model:
        print(f"\n🔍 Пример Word2Vec:")
        try:
            similar = embedding_trainer.word2vec_model.wv.most_similar("машинное", topn=3)
            for word, score in similar:
                print(f"  {word}: {score:.3f}")
        except KeyError:
            print("  Слово 'машинное' не найдено в словаре")


if __name__ == "__main__":
    main()