from util.dataset import make_dataset_for_classification
from util.decribe import get_unique_titles, get_unique_words_count, get_all_words, get_avg_word_len, \
    get_fraction_of_punctuation, get_fraction_of_upper, get_fraction_of_digit, get_all_categories, get_dataset

if __name__ == "__main__":
    data = get_dataset()

    print("Общее число документов: ", get_unique_titles(data))
    print("Уникальные слова: ", get_unique_words_count(data))
    print("Общее число слов: ", get_all_words(data))
    print("Среднее длина слов: ", get_avg_word_len())
    print("Доля знаков препинания: ", get_fraction_of_punctuation())
    print("Доля заглавных букв: ", get_fraction_of_upper())
    print("Доля цифр: ", get_fraction_of_digit())

    print("\n")
    print("Категории: ")
    for category in get_all_categories():
        print(category)

    dataset = make_dataset_for_classification()
    pos_class_count = 0
    neg_class_count = 0
    for item in dataset:
        if item['sentiment'] == 0:
            neg_class_count += 1
        else:
            pos_class_count += 1

    print("\n")
    print("Доля положительных новостей: ", pos_class_count / len(dataset))
    print("Доля отрицательных новостей: ", neg_class_count / len(dataset))