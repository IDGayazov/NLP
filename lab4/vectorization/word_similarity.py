import numpy as np

from gensim.models import Word2Vec, FastText
from typing import List, Tuple, Dict, Optional


class WordSimilarityFinder:
    """
    Класс для поиска похожих слов в обученных моделях
    """

    def __init__(self, model_path: Optional[str] = None, model_type: str = "word2vec"):
        """
        Args:
            model_path: путь к файлу модели
            model_type: тип модели ('word2vec' или 'fasttext')
        """
        self.model = None
        self.model_type = model_type

        if model_path:
            self.load_model(model_path, model_type)

    def load_model(self, model_path: str, model_type: str = "word2vec") -> None:
        """
        Загрузка обученной модели
        """
        try:
            if model_type == "word2vec":
                self.model = Word2Vec.load(model_path)
            elif model_type == "fasttext":
                self.model = FastText.load(model_path)
            else:
                raise ValueError("Поддерживаются только 'word2vec' и 'fasttext'")

            self.model_type = model_type
            print(f"✅ Модель {model_type} загружена из {model_path}")
            print(f"📊 Размер словаря: {len(self.model.wv)}")

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            self.model = None

    def find_similar_words(self, word: str, topn: int = 10, min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        """
        Поиск наиболее похожих слов

        Args:
            word: исходное слово
            topn: количество возвращаемых слов
            min_similarity: минимальная косинусная схожесть

        Returns:
            Список кортежей (слово, схожесть)
        """
        if self.model is None:
            print("❌ Модель не загружена!")
            return []

        try:
            similar_words = self.model.wv.most_similar(word, topn=topn)

            # Фильтруем по минимальной схожести
            filtered_words = [(w, score) for w, score in similar_words if score >= min_similarity]

            return filtered_words

        except KeyError:
            print(f"❌ Слово '{word}' не найдено в словаре модели")
            return []

    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Получение вектора слова
        """
        if self.model is None:
            return None

        try:
            return self.model.wv[word]
        except KeyError:
            print(f"❌ Слово '{word}' не найдено в словаре")
            return None

    def word_analogy(self, positive: List[str], negative: List[str], topn: int = 5) -> List[Tuple[str, float]]:
        """
        Решение задач аналогий: positive - negative = ?

        Пример: positive=['король', 'женщина'], negative=['мужчина']
                 результат ≈ 'королева'
        """
        if self.model is None:
            return []

        try:
            result = self.model.wv.most_similar(positive=positive, negative=negative, topn=topn)
            return result
        except Exception as e:
            print(f"❌ Ошибка в аналогии: {e}")
            return []

    def find_similar_multiple_models(self, word: str, model_paths: Dict[str, str], topn: int = 5) -> Dict[
        str, List[Tuple[str, float]]]:
        """
        Поиск похожих слов в нескольких моделях
        """
        results = {}

        for model_name, model_path in model_paths.items():
            print(f"\n🔍 Поиск в модели: {model_name}")

            # Определяем тип модели по имени файла
            model_type = "word2vec" if "word2vec" in model_path.lower() else "fasttext"

            try:
                # Временная загрузка модели
                if model_type == "word2vec":
                    temp_model = Word2Vec.load(model_path)
                else:
                    temp_model = FastText.load(model_path)

                similar_words = temp_model.wv.most_similar(word, topn=topn)
                results[model_name] = similar_words

                print(f"   Найдено {len(similar_words)} похожих слов")
                for w, score in similar_words[:3]:  # Покажем топ-3
                    print(f"   - {w}: {score:.3f}")

            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                results[model_name] = []

        return results


def interactive_similarity_search(model_path: str, model_type: str = "word2vec"):
    """
    Интерактивный поиск похожих слов
    """
    finder = WordSimilarityFinder(model_path, model_type)

    if finder.model is None:
        print("Не удалось загрузить модель!")
        return

    print(f"\n🎮 ИНТЕРАКТИВНЫЙ ПОИСК ПОХОЖИХ СЛОВ")
    print(f"Модель: {model_type}")
    print("Введите слово для поиска похожих (или 'quit' для выхода)")
    print("-" * 50)

    while True:
        user_input = input("\n🔍 Введите слово: ").strip()

        if user_input.lower() in ['quit', 'exit', 'выход']:
            print("👋 До свидания!")
            break

        if not user_input:
            continue

        # Поиск похожих слов
        similar_words = finder.find_similar_words(user_input, topn=8)

        if similar_words:
            print(f"\n📚 Слова похожие на '{user_input}':")
            for i, (word, similarity) in enumerate(similar_words, 1):
                print(f"  {i:2d}. {word:<15} (схожесть: {similarity:.3f})")
        else:
            print(f"😞 Не найдено похожих слов для '{user_input}'")


if __name__ == "__main__":
    # Если у вас есть реальные обученные модели, раскомментируйте:
    # interactive_similarity_search("../embeddings/word2vec_20251119_204821.model", "word2vec")
    interactive_similarity_search("../embeddings/fasttext_20251119_204821.model", "fasttext")