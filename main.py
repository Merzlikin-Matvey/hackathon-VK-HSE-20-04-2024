import os

import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification


def read_data(path="data.xlsx"):
    data = pd.read_excel(path, sheet_name=None)
    texts = []
    labels = []
    label_mapping = {sheet: i for i, sheet in enumerate(data)}
    for sheet in data:
        texts += data[sheet]['Title'].tolist()
        labels += [label_mapping[sheet]] * data[sheet].shape[0]
    return texts, labels, label_mapping

def print_predictions(model, tokenizer, dataset):
    for i, (inputs, label) in enumerate(dataset.take(5)):
        inputs = {k: v[None, :] for k, v in inputs.items()}
        prediction = model(inputs).logits
        print(f"Predictions: {prediction}")
        predicted_label = np.argmax(prediction, axis=1)[0]
        print(f"Text: {tokenizer.decode(inputs['input_ids'][0])}")
        print(f"True Label: {label.numpy()}")
        print(f"Predicted Label: {predicted_label}")
        print("\n")


if __name__ == '__main__':
    data = read_data()
    texts, labels, label_mapping = data

    if os.path.exists("model"):
        model = TFDistilBertForSequenceClassification.from_pretrained("model")
        tokenizer = DistilBertTokenizer.from_pretrained("model")

    else:
        model_name = "distilbert-base-uncased"
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(set(labels)))
        model.save_pretrained("model")
        tokenizer.save_pretrained("model")

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    train_encodings = tokenizer(X_train, truncation=True, padding=True)
    test_encodings = tokenizer(X_test, truncation=True, padding=True)

    train_dataset = tf.data.Dataset.from_tensor_slices((
        {key: np.array(value) for key, value in train_encodings.items()},
        y_train
    ))

    test_dataset = tf.data.Dataset.from_tensor_slices((
        {key: np.array(value) for key, value in test_encodings.items()},
        y_test
    ))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


    model.fit(train_dataset.shuffle(1000).batch(16),
                epochs=3,
                batch_size=16)

    print(print_predictions(model, tokenizer, test_dataset))


