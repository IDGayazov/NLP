from transformers import AutoTokenizer, AutoModel


class TatarTokenizer:
    def __init__(self):
        # Многоязычные модели, поддерживающие татарский
        self.models = {
            'bert-multilingual': 'bert-base-multilingual-cased',
            'xlm-roberta': 'xlm-roberta-base',
            'distilbert-multilingual': 'distilbert-base-multilingual-cased'
        }

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.models['xlm-roberta'])
            self.model = AutoModel.from_pretrained(self.models['xlm-roberta'])
            print("✅ Многоязычная модель XLM-RoBERTa загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")

    def tokenize_text(self, text, remove_special_chars=True):
        """Токенизация с помощью transformers"""
        tokens = self.tokenizer.tokenize(text)

        if remove_special_chars:
            # Убираем специальные символы ▁
            tokens = [token.replace('▁', '') for token in tokens if token not in ['<s>', '</s>', '<pad>']]
            # Фильтруем пустые токены
            tokens = [token for token in tokens if token and len(token) > 1]

        return tokens

    def get_token_ids(self, text):
        """Получение token IDs"""
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        return encoded

    def reconstruct_words(self, tokens):
        """Восстановление исходных слов из субтокенов"""
        words = []
        current_word = ""

        for token in tokens:
            if token.startswith('▁'):
                if current_word:
                    words.append(current_word)
                current_word = token[1:]  # Убираем ▁
            else:
                current_word += token

        if current_word:
            words.append(current_word)

        return words

if __name__ == "__main__":
    tatar_tokenizer = TatarTokenizer()
    # text = "Татар теле - матур тел"
    text = "Россия тимер юллары Яңа ел бәйрәмнәрендә 700дән артык өстәмә рейс җибәрәчәк"
    tokens = tatar_tokenizer.tokenize_text(text)
    print(tokens)

    words = tatar_tokenizer.reconstruct_words(tatar_tokenizer.tokenizer.tokenize(text))
    print("Восстановленные слова:", words)