import os.path

import numpy as np

from text_analytics import DATA, DATA_SIZE, get_most_important_words, convert_index_to_answer
from rnn import NeuralNetwork


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
        text = DATA['Article Text'][i]
        df = get_most_important_words(text)
        if len(df) < 10:
            continue
        else:
            df = df.head(10)
        X_sample = [df[column].values for column in ['Дети', 'Дом', 'Здоровье', 'Кино']]
        X.append(X_sample)
        y.append(get_index_for_category(DATA['Category'][i]))
    return np.array(X), np.array(y)


if __name__ == '__main__':
    load_learned = True
    model_path = "../model.keras"

    if load_learned and os.path.exists(model_path):
        nn = NeuralNetwork.load(model_path)
    else:
        X, y = prepare_data_for_nn(10000)
        nn = NeuralNetwork()
        nn.fit(X, y, epochs=10)
        nn.save(model_path)

    print("Model is ready!")
    X, y = prepare_data_for_nn(DATA_SIZE)
    nn = NeuralNetwork()


    num_of_test = 100
    print("TESTING")
    for i in range(num_of_test):
        rnd = np.random.randint(DATA_SIZE)
        predicted = nn.predict(X[rnd].reshape(1, 4, 10))
        print(nn.predict(X[rnd].reshape(1, 4, 10)))
        print()
        print()
