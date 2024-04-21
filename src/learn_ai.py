import os.path

import numpy as np

try:
    from text_analytics import DATA, DATA_SIZE, get_most_important_words, convert_index_to_answer
    from rnn import NeuralNetwork
except:
    from src.text_analytics import DATA, DATA_SIZE, get_most_important_words, convert_index_to_answer
    from src.rnn import NeuralNetwork


def get_index_for_category(key):
    return {
        "Ошибка" : -1,
        "Дети" : 0,
        "Дом" : 1,
        "Здоровье" : 2,
        "Кино" : 3
    }[key]


def prepare_data_for_nn(num=1000):
    X = []
    y = []
    for i in range(num):
        rnd = np.random.randint(DATA_SIZE)
        text = DATA['Article Text'][rnd]
        df = get_most_important_words(text)
        if len(df) < 10:
            continue
        else:
            df = df.head(10)
        X_sample = [df[column].values for column in ['Дети', 'Дом', 'Здоровье', 'Кино']]
        X.append(X_sample)
        good_ind = get_index_for_category(DATA['Category'][rnd])
        arr = [0, 0, 0, 0]
        arr[good_ind] = 1
        y.append(arr)
    return np.array(X), np.array(y)


def train(model, learning_data_size=6000):
    X, y = prepare_data_for_nn(learning_data_size)
    print(X.shape, y.shape)
    model.fit(X, y, epochs=100, batch_size=1500)
    return model


if __name__ == '__main__':
    model = NeuralNetwork()
    for i in range(6):
        model = train(model)

    print('Model trained')
    model.save("../model.keras")
    print('Model saved')

    df = get_most_important_words(
        "Фильм на вечер фильм катастрофа с неожиданным финалом очень большой взрыв который учит наслаждаться моментом")
    df_10 = df.head(10)
    X_sample = np.array([df_10[column].values for column in ['Дети', 'Дом', 'Здоровье', 'Кино']])
    X_sample = X_sample.reshape(1, 4, 10)
    prediction = model.predict(X_sample)
    print(X_sample, prediction)
    print(convert_index_to_answer(np.argmax(prediction)))


