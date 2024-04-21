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


def train(model, learning_data_size=1024*10):
    X, y = prepare_data_for_nn(learning_data_size)
    print(X.shape, y.shape)
    model.fit(X, y, epochs=50, batch_size=1024)
    return model


if __name__ == '__main__':
    model = NeuralNetwork()
    for i in range(4):
        model = train(model)
    print('Model trained')
    model.save("../model.keras")
    print('Model saved')


