from pkgutil import get_data

from sklearn.feature_extraction.text import TfidfVectorizer

from util.decribe import get_dataset


def create_tfidf_features(texts, max_features=5000):
    """
    Создает TF-IDF признаки из текстов
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    return vectorizer.fit_transform(texts), vectorizer

if __name__ == "__main__":
    data = get_dataset()

    texts = []
    for item in data:
        texts.append(item['text'])

    vec_texts, _ = create_tfidf_features(texts)