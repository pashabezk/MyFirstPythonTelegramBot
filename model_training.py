# модуль обучение модели

import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import strings as st

def train():
    config_file = open(st.BOT_DATASET_FOR_TRAINING_FILENAME, "r")
    BOT_CONFIG = json.load(config_file)

    X = []  # список заготовленных фраз из датасета, по которым  бот определяет intent (намерение)
    Y = []  # (классы) у каждой из заготовленных фраз уже определено намерение. Например у сообщения "Привет" намерение - приветствие (hello)
    # задача модели по Х находить Y

    # заполнение списков
    for name, data in BOT_CONFIG['intents'].items():
        for example in data['examples']:
            X.append(example)
            Y.append(name)

    # Векторизирование текстов: превращение их в цифровые данные - дать каждому слову свой номер
    vectorizer = CountVectorizer()
    vectorizer.fit(X)  # передаём набор текстjd, чтобы векторайзер их проанализировал
    # print(vectorizer.vocabulary_) #вывести список из пронумерованных строк

    X_vectorized = vectorizer.transform(X)  # трансформируем тексты в наборы чисел

    model = LogisticRegression()  # применяем модель обучения LogisticRegression
    model.fit(X_vectorized, Y)  # модель научится по Х понимать Y

    # проверка:
    # test = vectorizer.transform(["Как дела?"])
    # print(model.predict(test)) # по Х предсказать Y (по сообщению предсказать намерение (intent))

    # сохренение векторайзера в файл
    f = open(st.BOT_VECTORIZER_FILENAME, "wb")
    pickle.dump(vectorizer, f)

    # Сохранение модели в файл
    f = open(st.BOT_MODEL_FILENAME, "wb")
    pickle.dump(model, f)
