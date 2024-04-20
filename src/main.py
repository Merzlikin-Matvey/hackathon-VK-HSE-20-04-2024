from multiprocessing import Pool

import pandas as pd
import numpy as np

import re
import os
import time

data = pd.read_csv('../data/data.csv')
words_indexes = pd.read_csv('../data/words_indexes.csv')


def convert_index_to_answer(n):
    return {
        -1 : 'Ошибка',
        0 : 'Дети',
        1 : 'Дом',
        2 : 'Здоровье',
        3 : 'Кино'
    }[n]


def key_to_category(key):
    return {
         'kino.mail' : 'Кино',
         'deti.mail' : 'Дети',
         'dom.mail' : 'Дом',
         'health.mail' : 'Здоровье'
    }[key]


def get_words_stats(word):
    word = word.lower()
    if word not in words_indexes['word'].values:
        return None
    else:
        return words_indexes[words_indexes['word'] == word]


def get_texts_stats(text):
    text_stats = pd.DataFrame(columns=[*words_indexes.columns, 'count'])
    for word in re.findall(r"\w+", text):
        word_stats = get_words_stats(word)
        if word_stats is not None:
            if word not in text_stats.index:
                new_row = pd.concat([get_words_stats(word).reset_index(drop=True), pd.Series([1])], axis=1)
                new_row = new_row.rename(columns={0: 'count'})
                text_stats.loc[word] = new_row.loc[0]
            else:
                text_stats.loc[word, 'count'] += 1
    return text_stats.drop(columns='word')


def get_most_important_words(text):
    df = get_texts_stats(text)
    df['max_value'] = df[['Дети', 'Дом', 'Здоровье', 'Кино']].max(axis=1)
    df = df.sort_values(by='max_value', ascending=False)
    return df


def get_categories_probabilities(text):
    df = get_most_important_words(text)

    top_5 = df.head(5)
    values_5 = [top_5['Дети'].mean(), top_5['Дом'].mean(), top_5['Здоровье'].mean(), top_5['Кино'].mean()]
    return values_5


def get_category(text):
    probs = get_categories_probabilities(text)
    rounded_probs = [round(prob, 4) for prob in probs]
    print(rounded_probs)
    if max(probs) > 0.45:
        return convert_index_to_answer(probs.index(max(probs)))
    else:
        return 'Дети'


def evaluate_sample(i):
    predicted = get_category(data['Article Text'][i])
    real = data['Category'][i]
    return predicted == real


def test(n=50):
    correct = 0
    all = 0

    indices = [np.random.randint(0, len(data)) for _ in range(n)]

    with Pool(os.cpu_count()) as p:
        results = p.map(evaluate_sample, indices)

    for result in results:
        all += 1
        correct += result

    return [correct, all]

if __name__ == '__main__':
    n = 5 * 12
    start = time.time()
    print(test(n))
    end = time.time()
    print(f"Time: {end - start}")
    print(f"Time per sample: {(end - start) / n}")