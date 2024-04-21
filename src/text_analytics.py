from multiprocessing import Pool
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

import pandas as pd
import numpy as np

import re
import os
import time

if os.path.exists('data/data.csv'):
    DATA = pd.read_csv('data/data.csv')
    WORDS_INDEXES = pd.read_csv('data/WORDS_INDEXES.csv').set_index('word').T.to_dict('list')
else:
    DATA = pd.read_csv('../data/data.csv')
    WORDS_INDEXES = pd.read_csv('../data/WORDS_INDEXES.csv').set_index('word').T.to_dict('list')


DATA_SIZE = len(DATA)

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
    return WORDS_INDEXES.get(word, None)


def get_texts_stats(text):
    text_stats = []
    for word in re.findall(r"\w+", text):
        if len(word) <= 2:
            continue
        word_stats = get_words_stats(word)
        if word_stats is not None:
            if not any(d['word'] == word for d in text_stats):
                new_row = {'word': word, 'Дети': word_stats[0], 'Дом': word_stats[1], 'Здоровье': word_stats[2], 'Кино': word_stats[3], 'count': 1}
                text_stats.append(new_row)
            else:
                for d in text_stats:
                    if d['word'] == word:
                        d['count'] += 1
                        break
    return pd.DataFrame(text_stats).set_index('word')


def get_most_important_words(text):
    df = get_texts_stats(text)
    df['max_value'] = df[['Дети', 'Дом', 'Здоровье', 'Кино']].max(axis=1)
    df = df.sort_values(by='max_value', ascending=False)
    return df

def get_category(text):
    df = get_most_important_words(text)

    if len(df) <= 10:
        top_5 = df.head(5)
        values_5 = [top_5['Дети'].mean(), top_5['Дом'].mean(), top_5['Здоровье'].mean(), top_5['Кино'].mean()]
        return convert_index_to_answer(np.argmax(values_5))
    else:
        pass




def evaluate_sample(i):
    predicted = get_category(DATA['Article Text'][i])
    real = DATA['Category'][i]
    return predicted, real


def test(n=60, category=None):
    global data
    y_true = []
    y_pred = []


    indices = [np.random.randint(0, DATA_SIZE) for _ in range(n)]

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

