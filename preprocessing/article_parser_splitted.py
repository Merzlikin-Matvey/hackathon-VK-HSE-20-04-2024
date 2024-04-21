# HARDCODED PART - gain data from xlsx
import pandas as pd
import numpy as np


sheet_to_category = {
    "kino.mail": "Кино",
    "health.mail": "Здоровье",
    "dom.mail": "Дом",
    "deti.mail": "Дети",
}
out_test = pd.DataFrame(columns=["Class", "Title", "Description", "Text"])
out_train = pd.DataFrame(columns=["Class", "Title", "Description", "Text"])
for sheet, category in sheet_to_category.items():
    df = pd.read_excel("data.mail.xlsx", sheet)
    df["Class"] = category
    df["Text"] = df["Article Text"]
    df = df[["Class", "Title", "Description", "Text"]]
    to_test_indexes = np.random.default_rng().choice(
        len(df), int(len(df) * 0.1), replace=False
    )
    to_train_indexes = np.delete(np.arange(len(df)), to_test_indexes)
    out_test = pd.concat([out_test, df.iloc[to_test_indexes]])
    out_train = pd.concat([out_train, df.iloc[to_train_indexes]])
out_test.to_csv("data_processed_test.csv", index=False)
out_train.to_csv("data_processed_train.csv", index=False)
