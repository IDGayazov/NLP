# text_preprocessing.py
import numpy as np
import pandas as pd

from typing import List, Dict, Union
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from preprocessing.text_preprocessing import preprocess
from util.jsonl_process import read_jsonl_basic
from util.load_tokenize_dataset import load_tokenize_ds


class StatisticalVectorizer:
    """
    Класс для статистической векторизации текстов
    Поддерживает как сплошной текст, так и токенизированный текст
    """

    def __init__(self, max_features: int = 5000, min_df: int = 2, max_df: float = 0.8):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df

        # Инициализация векторзаторов
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None
        self.ngram_vectorizer = None

        self.vocabulary_ = None

    def _prepare_texts(self, texts: List[Union[str, List[str]]]) -> List[str]:
        """
        Подготовка текстов: конвертация токенизированного текста в строки

        Args:
            texts: список текстов (строк или списков токенов)

        Returns:
            Список строк, готовых для векторизации
        """
        if not texts:
            return texts

        # Проверяем тип первого элемента
        if isinstance(texts[0], list):
            # Это токенизированный текст - объединяем токены в строки
            return [' '.join(tokens) for tokens in texts]
        elif isinstance(texts[0], str):
            # Уже строки - возвращаем как есть
            return texts
        else:
            raise ValueError(f"Неподдерживаемый тип текстов: {type(texts[0])}")

    def fit_bow(self, texts: List[Union[str, List[str]]]) -> None:
        """
        Обучение Bag of Words векторзатора

        Args:
            texts: список текстов для обучения (строк или списков токенов)
        """
        prepared_texts = self._prepare_texts(texts)

        self.bow_vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=None,  # Стоп-слова уже удалены на этапе препроцессинга
            lowercase=False,  # Текст уже в нижнем регистре
            tokenizer=lambda x: x.split(),  # Используем простой сплит для предобработанных текстов
            preprocessor=None  # Текст уже предобработан
        )

        self.bow_vectorizer.fit(prepared_texts)
        self.vocabulary_ = self.bow_vectorizer.vocabulary_

    def transform_bow(self, texts: List[Union[str, List[str]]]) -> np.ndarray:
        """
        Преобразование текстов в BoW векторы

        Args:
            texts: список текстов (строк или списков токенов)

        Returns:
            Массив векторов формы (n_documents, n_features)
        """
        if self.bow_vectorizer is None:
            raise ValueError("Сначала обучите модель с помощью fit_bow()")

        prepared_texts = self._prepare_texts(texts)
        return self.bow_vectorizer.transform(prepared_texts).toarray()

    def get_bow_features(self, texts: List[Union[str, List[str]]]) -> Dict:
        """
        Получение BoW представления с метаинформацией

        Args:
            texts: список текстов (строк или списков токенов)
        """
        vectors = self.transform_bow(texts)

        return {
            'vectors': vectors,
            'vocabulary': self.vocabulary_,
            'feature_names': self.bow_vectorizer.get_feature_names_out(),
            'vector_type': 'bow'
        }

    def fit_tfidf(self, texts: List[Union[str, List[str]]], use_idf: bool = True) -> None:
        """
        Обучение TF-IDF векторзатора

        Args:
            texts: список текстов для обучения (строк или списков токенов)
            use_idf: использовать ли обратную частоту документа
        """
        prepared_texts = self._prepare_texts(texts)

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            use_idf=use_idf,
            norm='l2',  # Нормализация L2
            stop_words=None,
            lowercase=False,
            tokenizer=lambda x: x.split(),
            preprocessor=None
        )

        self.tfidf_vectorizer.fit(prepared_texts)
        self.vocabulary_ = self.tfidf_vectorizer.vocabulary_

    def transform_tfidf(self, texts: List[Union[str, List[str]]]) -> np.ndarray:
        """
        Преобразование текстов в TF-IDF векторы

        Args:
            texts: список текстов (строк или списков токенов)
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("Сначала обучите модель с помощью fit_tfidf()")

        prepared_texts = self._prepare_texts(texts)
        return self.tfidf_vectorizer.transform(prepared_texts).toarray()

    def get_tfidf_features(self, texts: List[Union[str, List[str]]]) -> Dict:
        """
        Получение TF-IDF представления

        Args:
            texts: список текстов (строк или списков токенов)
        """
        vectors = self.transform_tfidf(texts)

        # Получаем IDF веса если они используются
        idf_weights = None
        if hasattr(self.tfidf_vectorizer, 'idf_'):
            idf_weights = dict(zip(
                self.tfidf_vectorizer.get_feature_names_out(),
                self.tfidf_vectorizer.idf_
            ))

        return {
            'vectors': vectors,
            'vocabulary': self.vocabulary_,
            'feature_names': self.tfidf_vectorizer.get_feature_names_out(),
            'idf_weights': idf_weights,
            'vector_type': 'tfidf'
        }

    def fit_ngrams(self, texts: List[Union[str, List[str]]], ngram_range: tuple = (1, 2),
                   method: str = 'tfidf') -> None:
        """
        Обучение N-gram векторзатора

        Args:
            texts: список текстов для обучения (строк или списков токенов)
            ngram_range: диапазон N-грамм (1,2) = униграммы + биграммы
            method: 'bow' или 'tfidf'
        """
        prepared_texts = self._prepare_texts(texts)

        if method == 'tfidf':
            self.ngram_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                ngram_range=ngram_range,
                stop_words=None,
                lowercase=False,
                tokenizer=lambda x: x.split(),
                preprocessor=None
            )
        else:  # 'bow'
            self.ngram_vectorizer = CountVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                ngram_range=ngram_range,
                stop_words=None,
                lowercase=False,
                tokenizer=lambda x: x.split(),
                preprocessor=None
            )

        self.ngram_vectorizer.fit(prepared_texts)
        self.vocabulary_ = self.ngram_vectorizer.vocabulary_

    def transform_ngrams(self, texts: List[Union[str, List[str]]]) -> np.ndarray:
        """
        Преобразование текстов в N-gram векторы

        Args:
            texts: список текстов (строк или списков токенов)
        """
        if self.ngram_vectorizer is None:
            raise ValueError("Сначала обучите модель с помощью fit_ngrams()")

        prepared_texts = self._prepare_texts(texts)
        return self.ngram_vectorizer.transform(prepared_texts).toarray()

    def get_ngram_features(self, texts: List[Union[str, List[str]]]) -> Dict:
        """
        Получение N-gram представления

        Args:
            texts: список текстов (строк или списков токенов)
        """
        vectors = self.transform_ngrams(texts)

        feature_names = self.ngram_vectorizer.get_feature_names_out()

        # Анализ N-грамм
        ngram_analysis = self._analyze_ngrams(feature_names)

        return {
            'vectors': vectors,
            'vocabulary': self.vocabulary_,
            'feature_names': feature_names,
            'ngram_analysis': ngram_analysis,
            'vector_type': 'ngram'
        }

    def _analyze_ngrams(self, feature_names: List[str]) -> Dict:
        """
        Анализ распределения N-грамм
        """
        unigrams = [f for f in feature_names if len(f.split()) == 1]
        bigrams = [f for f in feature_names if len(f.split()) == 2]
        trigrams = [f for f in feature_names if len(f.split()) == 3]

        return {
            'unigrams_count': len(unigrams),
            'bigrams_count': len(bigrams),
            'trigrams_count': len(trigrams),
            'total_ngrams': len(feature_names),
            'sample_unigrams': unigrams[:10],
            'sample_bigrams': bigrams[:10],
            'sample_trigrams': trigrams[:10] if trigrams else []
        }

    def fit_all(self, texts: List[Union[str, List[str]]], ngram_range: tuple = (1, 2)) -> None:
        """
        Обучение всех векторзаторов одновременно

        Args:
            texts: список текстов (строк или списков токенов)
            ngram_range: диапазон N-грамм
        """
        print("Обучение BoW векторзатора...")
        self.fit_bow(texts)

        print("Обучение TF-IDF векторзатора...")
        self.fit_tfidf(texts)

        print("Обучение N-gram векторзатора...")
        self.fit_ngrams(texts, ngram_range, method='tfidf')

    def get_all_features(self, texts: List[Union[str, List[str]]]) -> Dict[str, Dict]:
        """
        Получение всех типов векторизаций

        Args:
            texts: список текстов (строк или списков токенов)
        """
        features = {}

        try:
            features['bow'] = self.get_bow_features(texts)
        except ValueError as e:
            print(f"BoW features not available: {e}")

        try:
            features['tfidf'] = self.get_tfidf_features(texts)
        except ValueError as e:
            print(f"TF-IDF features not available: {e}")

        try:
            features['ngram'] = self.get_ngram_features(texts)
        except ValueError as e:
            print(f"N-gram features not available: {e}")

        return features

    def analyze_features(self, texts: List[Union[str, List[str]]], top_n: int = 20) -> Dict:
        """
        Анализ наиболее важных признаков

        Args:
            texts: список текстов (строк или списков токенов)
            top_n: количество топ-признаков для анализа
        """
        features = self.get_all_features(texts)
        analysis = {}

        for method, feature_data in features.items():
            vectors = feature_data['vectors']
            feature_names = feature_data['feature_names']

            # Средние веса по всем документам
            mean_weights = np.mean(vectors, axis=0)

            # Топ-N наиболее важных признаков
            top_indices = np.argsort(mean_weights)[-top_n:][::-1]
            top_features = [
                (feature_names[i], mean_weights[i])
                for i in top_indices
            ]

            analysis[method] = {
                'top_features': top_features,
                'feature_count': len(feature_names),
                'sparsity': np.mean(vectors == 0)  # Спарсность матрицы
            }

        return analysis

    def feature_statistics(self, texts: List[Union[str, List[str]]]) -> pd.DataFrame:
        """
        Статистика по признакам в виде DataFrame

        Args:
            texts: список текстов (строк или списков токенов)
        """
        features = self.get_all_features(texts)
        stats = []

        for method, feature_data in features.items():
            vectors = feature_data['vectors']

            stats.append({
                'method': method.upper(),
                'feature_count': vectors.shape[1],
                'document_count': vectors.shape[0],
                'sparsity': f"{np.mean(vectors == 0) * 100:.1f}%",
                'mean_non_zero': np.mean(vectors[vectors != 0]),
                'std_non_zero': np.std(vectors[vectors != 0])
            })

        return pd.DataFrame(stats)

    def get_input_type(self, texts: List[Union[str, List[str]]]) -> str:
        """
        Определяет тип входных данных

        Returns:
            'tokens' или 'text'
        """
        if not texts:
            return 'empty'

        if isinstance(texts[0], list):
            return 'tokens'
        elif isinstance(texts[0], str):
            return 'text'
        else:
            return 'unknown'


def demonstrate_vectorization():
    # sample_texts = [
    #     "машинное обучение искусственный интеллект",
    #     "нейронные сети глубокое обучение",
    #     "обработка естественного языка NLP",
    #     "компьютерное зрение машинное обучение",
    #     "татарский язык лингвистика обработка текста"
    # ]

    texts = load_tokenize_ds()

    # Инициализация векторзатора
    vectorizer = StatisticalVectorizer(max_features=1000, min_df=1)

    # Обучение на всех методах
    vectorizer.fit_all(texts)

    # Получение всех признаков
    all_features = vectorizer.get_all_features(texts)

    # Анализ
    analysis = vectorizer.analyze_features(texts)
    stats_df = vectorizer.feature_statistics(texts)

    print("=== СТАТИСТИКА ПРИЗНАКОВ ===")
    print(stats_df)

    print("\n=== ТОП ПРИЗНАКИ TF-IDF ===")
    for feature, weight in analysis['tfidf']['top_features'][:10]:
        print(f"{feature}: {weight:.4f}")

    return all_features


if __name__ == "__main__":
    features = demonstrate_vectorization()