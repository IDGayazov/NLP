from typing import List


class TatarStopWordsRemover:
    def __init__(self):
        # Базовый список татарских стоп-слов
        self.tatar_stopwords = {
            'һәм', 'вә', 'белән', 'өчен', 'турында', 'буенча', 'дип', 'бу', 'да',
            'әле', 'инде', 'мени', 'ни', 'кем', 'кайдан', 'никадәр',
            'ничек', 'нәрсә', 'кайчан', 'кайда', 'күпме', 'ничә',
            'ул', 'без', 'сез', 'алар', 'мин', 'син', 'улар',
            'монда', 'анда', 'шунда', 'соң', 'әүвәл', 'лә', 'дә',
            'та', 'тә', 'ны', 'не', 'на', 'нә', 'нын', 'нен'
        }

    def remove_tatar_stopwords(self, tokens: List[str]) -> List[str]:
        """Удаляет татарские стоп-слова"""
        return [token for token in tokens
                if token.lower() not in self.tatar_stopwords and len(token) > 1]

if __name__ == "__main__":
    stop_words_remover = TatarStopWordsRemover()

    text = "Россия тимер юллары Яңа ел һәм бәйрәмнәрендә 700дән артык өстәмә рейс җибәрәчәк"

    words = []
    for word in text.split(' '):
        words.append(word)

    print(stop_words_remover.remove_tatar_stopwords(words))