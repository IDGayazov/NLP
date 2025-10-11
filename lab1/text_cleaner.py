import re
import html
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextCleaner:
    def __init__(self, lowercase=True, remove_stopwords=True, language='russian'):
        """
        Инициализация очистителя текста

        Args:
            lowercase: Приводить текст к нижнему регистру
            remove_stopwords: Удалять стоп-слова
            language: Язык для стоп-слов ('russian', 'tatar', 'english')
        """
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.language = language

        # Загружаем стоп-слова
        if self.remove_stopwords:
            self.stop_words = self._load_stopwords(language)

    def _load_stopwords(self, language):
        """Загрузка стоп-слов для разных языков"""
        if language == 'russian':
            try:
                return set(stopwords.words('russian'))
            except:
                # Резервный список русских стоп-слов
                return self._get_russian_stopwords()

        elif language == 'tatar':
            # Татарские стоп-слова
            return self._get_tatar_stopwords()

        elif language == 'english':
            return set(stopwords.words('english'))

        else:
            print(f"Язык '{language}' не поддерживается. Используем русские стоп-слова.")
            return self._get_russian_stopwords()

    def _get_russian_stopwords(self):
        """Русские стоп-слова"""
        return {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а',
            'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
            'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
            'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже',
            'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него',
            'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом',
            'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо',
            'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без',
            'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда',
            'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним',
            'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас',
            'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец',
            'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через',
            'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три',
            'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда',
            'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда',
            'конечно', 'всю', 'между'
        }

    def _get_tatar_stopwords(self):
        """Татарские стоп-слова"""
        return {
            'вә', 'һәм', 'белән', 'өчен', 'турында', 'карата', 'буенча', 'аркылы',
            'мин', 'син', 'ул', 'без', 'сез', 'алар', 'монда', 'анда', 'шунда',
            'ни', 'нәрсә', 'кем', 'кайсы', 'ничек', 'никадәр', 'кайчан', 'нишләп',
            'әле', 'инде', 'һаман', 'тагын', 'күбрәк', 'азрак', 'бик', 'бигерәк',
            'тә', 'дә', 'мы', 'ме', 'бы', 'бе', 'гы', 'ге', 'ка', 'кә',
            'ук', 'генә', 'гына', 'чик', 'хәтта', 'әллә', 'гомер', 'көн', 'ел',
            'яки', 'ягъни', 'яисә', 'әгәр', 'чиң', 'шул', 'бу', 'бер', 'ике', 'өч',
            'үз', 'башка', 'барлык', 'бөтен', 'һич', 'бернинди', 'беркайчан',
            'бар', 'юк', 'әйе', 'юк', 'шулай', 'түгел', 'мөмкин', 'кирәк', 'була',
            'дип', 'ди', 'дә', 'тә', 'мыни', 'әйтә', 'күрсәтә', 'алына', 'бирелеп'
        }

    def clean_html(self, text):
        """Удаление HTML-разметки"""
        if not text:
            return ""

        text = html.unescape(text)
        soup = BeautifulSoup(text, 'html.parser')
        clean_text = soup.get_text(separator=' ')
        return clean_text

    def remove_special_characters(self, text):
        """Удаление служебных символов"""
        if not text:
            return ""

        # Удаляем email и URL
        text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\S+', '', text)

        # Для татарского сохраняем специфические символы
        if self.language == 'tatar':
            text = re.sub(r'[^\w\s\.\,\!\?\-\:\(\)әөүҗңһ]', '', text)
        else:
            text = re.sub(r'[^\w\s\.\,\!\?\-\:\(\)]', '', text)

        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def normalize_whitespace(self, text):
        """Стандартизация пробельных символов"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def remove_stopwords_func(self, text):
        """Фильтрация стоп-слов"""
        if not text or not self.remove_stopwords:
            return text

        # Простая токенизация (можно заменить на более сложную)
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]

        return ' '.join(filtered_words)

    def clean_text(self, text, **kwargs):
        """Основная функция очистки текста"""
        if not text:
            return ""

        cleaned_text = text

        # Применяем очистку HTML
        if kwargs.get('clean_html', True):
            cleaned_text = self.clean_html(cleaned_text)

        # Удаляем специальные символы
        if kwargs.get('remove_special_chars', True):
            cleaned_text = self.remove_special_characters(cleaned_text)

        # Нормализуем пробелы
        if kwargs.get('normalize_whitespace', True):
            cleaned_text = self.normalize_whitespace(cleaned_text)

        # Приводим к нижнему регистру
        if kwargs.get('lowercase', self.lowercase):
            cleaned_text = cleaned_text.lower()

        # Удаляем стоп-слова
        if kwargs.get('remove_stopwords', self.remove_stopwords):
            cleaned_text = self.remove_stopwords_func(cleaned_text)

        return cleaned_text


# Пример использования с татарским языком
if __name__ == "__main__":
    # Татарский текст
    tatar_text = "Мин <p> һәм син бу көнне бик яхшы вакыт үткәрдек<p>.5 Ул безгә киләчәк."

    cleaner_ru = TextCleaner(language='russian')
    cleaner_tt = TextCleaner(language='tatar')

    print("Татарский текст:")
    print(tatar_text)
    print("\nОчистка как русский текст:")
    print(cleaner_ru.clean_text(tatar_text))
    print("\nОчистка как татарский текст:")
    print(cleaner_tt.clean_text(tatar_text))