import re


class TatarLemmatizer:
    def __init__(self):
        """
        Лемматизатор для татарского языка (rule-based подход)
        """
        # Словарь базовых форм
        self.tatar_word_forms = {
            'тел': 'тел',
            'теле': 'тел',
            'телен': 'тел',
            'телебез': 'тел',
            'матур': 'матур',
            'матуры': 'матур',
            'матурлык': 'матурлык',
            'казан': 'казан',
            'казанны': 'казан',
            'казанда': 'казан'
        }

        # Суффиксы для удаления
        self.suffixes = [
            'ны', 'не', 'на', 'не', 'нын', 'нен', 'нан', 'нән',  # падежи
            'лар', 'ләр', 'нар', 'нәр', 'лары', # множественное число
            'ым', 'ем', 'ың', 'ең', 'ыбыз', 'ебез',  # притяжательность
            'ык', 'ек', 'ук', 'үк', 'кы', 'ке', 'чәк', 'чак', 'нәрендә'  # производные
        ]

    def rule_based_lemmatize(self, word: str) -> str:
        """
        Rule-based лемматизация для татарского языка
        """
        # Сначала проверяем словарь
        if word in self.tatar_word_forms:
            return self.tatar_word_forms[word]

        # Пробуем удалить суффиксы
        for suffix in self.suffixes:
            if word.endswith(suffix) and len(word) >= len(suffix) + 2:
                base_form = word[:-len(suffix)]
                if base_form in self.tatar_word_forms:
                    return self.tatar_word_forms[base_form]
                else:
                    return base_form

        return word

    def lemmatize_tatar_text(self, text: str):
        """
        Лемматизация татарского текста
        """
        # Простая токенизация
        tokens = re.findall(r'[а-яәөүҗңһ]+', text.lower())
        lemmas = []

        for token in tokens:
            lemma = self.rule_based_lemmatize(token)
            if len(lemma) > 1:
                lemmas.append(lemma)

        return lemmas

if __name__ == "__main__":
    text = "Россия тимер юллары Яңа ел бәйрәмнәрендә 700дән артык өстәмә рейс җибәрәчәк"
    lemmatizer = TatarLemmatizer()

    words = []
    for word in text.split(' '):
        words.append(lemmatizer.rule_based_lemmatize(word))

    print(words)