# Бот для телеграмма с ИИ

Бот разработан в процессе прохождения трёхдневного интенсива «Чат-бот с искусственным интеллектом на Python» от Skillbox.

Бот обучен по датасету `dataset_for_bot.json`, предоставленным в процессе прохождения интенсива.

Структура данных, по которым обучается бот, выглядит следующим образом:
``` python
BOT_CONFIG = {
    # Намерения пользователя
    "intents": {
        # желание поздороваться
        "hello" : {
            "examples" : ["Привет", "Здарова", "Добрый день", "Здрасте", "Здравствуйте", "Доброго времени суток"],
            "responses" : ["И тебе привет, человек", "Привки", "Привет"]
        },
        # прощание
        "bye" : {
            "examples" : ["Пока", "До свидания", "Досвидос", "Покасики", "Прощай", "Покедова"],
            "responses" : ["До скорых встреч!", "До свидания", "Пиши, если что"]
        },
        # как дела
        "how are you" : {
            "examples" : ["Как дела", "Как делишки", "Че как", "Что делаешь", "Чем занят"],
            "responses" : ["Отвечаю на глупые вопросы", "Смотрю видосики", "Не разговариваю с тобой"]
        }
    },
    "failure_phrases" : ["Ничего не понял", "Я не знаю такого", "Спроси что-нибудь попроще"]
}
```

В общих чертах:
При получении сообщения (примеры сообщений - `examples`), бот пытается определить намерение (`intent`) и выдаёт один из запрограммированных вариантов ответа (`responses`).

Подробно:
1. Полученный текст сообщения приводится к нижнему регистру, а также отбрасываются все лишние символы, кроме букв и пробелов.
2. Далее с помощью функции `nltk.edit_distance(text1, text2)` ищется разница в количестве букв между полученным сообщением с запрограммированными вариантами (`examples`). Количество букв нормализируется относительно средней длины текстов, и если они совпадают не менее, чем на 40 процентов, то считается, что намерение определено.
3. Если намерение определить не удалось, то запускается обученная модель. На основе векторизации текстов она пытается подобрать намерение.
4. По определенному намерению выдается один из запрограммированных ответов.


## Структура проекта

В папке Files расположены:
* `dataset_for_bot.json` - набор данных для обучения бота;
* `bot_model.bin` - обученная по датасету модель;
* `bot_vectorizer.bin` - векторайзер для работы модели.

В корневой папке:
* `main.py` - основной файл;
* `model_training.py` - модуль для обучения модели по датасету, запускается только если отсутствует файлы `bot_model.bin`, `bot_vectorizer.bin`;
* `strings.py` - переменные, в которых хранятся пути к файлам.

## Установка и настройка

Для работы бота нужен токен. Чтобы его получить, необходимо написать телеграмм-боту [@BotFather](https://t.me/BotFather).
1) `/newbot` - команда для регистрации бота в тг
2) после этого будет предложено написать имя бота, например `MyFirstPythonBot`
3) дальше нужно выбрать username бота, то есть ссылку, по которой он будет доступен, например `MyFirstPython_bot` (такое имя выбрать не получится, т.к. оно уже занято, необходимо придумать уникальный username)

Если все действия были выполнены правильно, то [@BotFather](https://t.me/BotFather) пришлет текст, который будет содержать токен для работы бота:
`Use this token to access the HTTP API: <токен для работы бота>`

Далее создайте в папке проекта файл `token.txt` и поместите туда токен для работы телеграм бота.
> P.s. при сохранении на GitHub файл `token.txt` загружаться не будет, так как он добавлен в `.gitignore`. Это сделано для безопасности, чтобы никто не мог воспользоваться Вашим ботом.

Также для работы бота необходимо иметь установленную библиотеку [python-telegram-bot](https://python-telegram-bot.org/). Для её установки воспользуйтесь командой:
`pip install python-telegram-bot`