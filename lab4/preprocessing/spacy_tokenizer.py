import spacy

from typing import List, Dict, Any


class SpacyTokenizer:
    def __init__(self, model_name: str = "ru_core_news_sm"):
        """
        Инициализация токенизатора spaCy

        Args:
            model_name: название модели для русского языка
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"❌ Модель {model_name} не найдена. Скачиваем...")
            from spacy.cli import download
            download(model_name)
            self.nlp = spacy.load(model_name)

    def tokenize_text(self, text: str, remove_punctuation: bool = True,
                      remove_stopwords: bool = False, lemmatize: bool = True) -> List[str]:
        """
        Токенизация текста с различными опциями

        Args:
            text: исходный текст
            remove_punctuation: удалять пунктуацию
            remove_stopwords: удалять стоп-слова
            lemmatize: проводить лемматизацию

        Returns:
            Список токенов
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
        tokens = []

        for token in doc:
            # Пропускаем пробелы
            if token.is_space:
                continue

            # Пропускаем пунктуацию если нужно
            if remove_punctuation and token.is_punct:
                continue

            # Пропускаем стоп-слова если нужно
            if remove_stopwords and token.is_stop:
                continue

            # Выбираем лемму или оригинальный текст
            if lemmatize and token.lemma_.strip():
                token_text = token.lemma_.lower().strip()
            else:
                token_text = token.text.lower().strip()

            if token_text:
                tokens.append(token_text)

        return tokens

    def get_token_details(self, text: str) -> List[Dict[str, Any]]:
        """
        Получение детальной информации о каждом токене

        Returns:
            Список словарей с информацией о токенах
        """
        if not text:
            return []

        doc = self.nlp(text)
        token_info = []

        for token in doc:
            token_info.append({
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'tag': token.tag_,
                'dep': token.dep_,
                'is_alpha': token.is_alpha,
                'is_stop': token.is_stop,
                'is_punct': token.is_punct,
                'is_space': token.is_space,
                'is_digit': token.is_digit
            })

        return token_info


if __name__ == "__main__":
    tokenizer = SpacyTokenizer()

    # Пример 1: Базовая токенизация
    text = "Машинное обучение — это раздел искусственного интеллекта."
    tokens = tokenizer.tokenize_text(text)
    print("Базовые токены:", tokens)

    text1 = "Россия тимер юллары Яңа ел бәйрәмнәрендә 700дән артык өстәмә рейс җибәрәчәк"
    tokens1 = tokenizer.tokenize_text(text1)
    print("Базовые токены:", tokens1)