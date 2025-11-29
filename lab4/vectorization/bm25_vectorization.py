from rank_bm25 import BM25Okapi
import numpy as np

from util.decribe import get_dataset


class BM25VectorizerShort:
    """
    Короткая реализация BM25 векторизации
    """

    def __init__(self, k1=1.5, b=0.75):
        self.bm25 = None
        self.k1 = k1
        self.b = b
        self.vocabulary_ = None

    def fit_transform(self, texts):
        """Обучение и преобразование текстов в BM25 матрицу"""
        # Токенизация текстов
        tokenized_texts = [text.split() for text in texts]

        # Создаем словарь
        self.vocabulary_ = set()
        for tokens in tokenized_texts:
            self.vocabulary_.update(tokens)
        self.vocabulary_ = list(self.vocabulary_)

        # Обучаем BM25
        self.bm25 = BM25Okapi(tokenized_texts, k1=self.k1, b=self.b)

        # Преобразуем в матрицу scores
        scores = []
        for tokens in tokenized_texts:
            doc_scores = self.bm25.get_scores(tokens)
            scores.append(doc_scores)

        return np.array(scores)

    def transform(self, texts):
        """Преобразование новых текстов в BM25 матрицу"""
        if self.bm25 is None:
            raise ValueError("Сначала вызовите fit_transform!")

        tokenized_texts = [text.split() for text in texts]

        scores = []
        for tokens in tokenized_texts:
            doc_scores = self.bm25.get_scores(tokens)
            scores.append(doc_scores)

        return np.array(scores)

    def get_feature_names(self):
        """Получить названия признаков (токены)"""
        return self.vocabulary_ if self.vocabulary_ is not None else []

    def get_vocabulary_size(self):
        """Получить размер словаря"""
        return len(self.vocabulary_) if self.vocabulary_ is not None else 0


if __name__ == "__main__":
    data = get_dataset()

    texts = []
    for item in data:
        texts.append(item['text'])

    # Создаем и обучаем BM25
    bm25 = BM25VectorizerShort()
    X = bm25.fit_transform(texts)

    print(f"📊 Размерность: {X.shape}")
    print(f"🔤 Размер словаря: {bm25.get_vocabulary_size()}")
    print(f"📝 Примеры признаков: {bm25.get_feature_names()[:10]}")