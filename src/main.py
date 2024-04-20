from multiprocessing import Pool
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

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


def get_category(text):
    df = get_most_important_words(text)

    top_5 = df.head(5)
    values_5 = [top_5['Дети'].mean(), top_5['Дом'].mean(), top_5['Здоровье'].mean(), top_5['Кино'].mean()]

    if max(values_5) > 0.7:
        return convert_index_to_answer(values_5.index(max(values_5)))
    else:
        tmp = 0
        for i in range(5):
            if top_5['Дети'].iloc[i] > 0.85:
                tmp += 2
        if tmp >= 2:
            return 'Дети'
        else:
            top_20 = df.head()
            values_20 = [top_20['Дети'].mean(), top_20['Дом'].mean(), top_20['Здоровье'].mean(), top_20['Кино'].mean()]
            return convert_index_to_answer(values_20.index(max(values_20)))


def evaluate_sample(i):
    predicted = get_category(data['Article Text'][i])
    real = data['Category'][i]
    return predicted, real


def test(n=50):
    y_true = []
    y_pred = []

    indices = [np.random.randint(0, len(data)) for _ in range(n)]

    with Pool(os.cpu_count()) as p:
        for predicted, real in tqdm(p.imap(evaluate_sample, indices), total=n):
            y_pred.append(predicted)
            y_true.append(real)


    f1 = f1_score(y_true, y_pred, average='weighted')

    accuracy = sum([1 for i in range(n) if y_true[i] == y_pred[i]]) / n

    lb = LabelBinarizer()
    y_true = lb.fit_transform(y_true)
    y_pred = lb.transform(y_pred)
    auc = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')

    return accuracy, f1, auc


if __name__ == '__main__':
    n = 10 * os.cpu_count()
    print(f"Testing on {n} samples")
    print(f"Number of CPU: {os.cpu_count()}")
    start = time.time()
    accuracy, f1, auc = test(n)
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"AUC Score: {auc}")
    end = time.time()
    print(f"Time: {end - start}")
    print(f"Time per sample: {(end - start) / n}")