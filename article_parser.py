# HARDCODED PART - gain data from xlsx
import pandas as pd


sheet_to_category = {
    "kino.mail": "Кино",
    "health.mail": "Здоровье",
    "dom.mail": "Дом",
    "deti.mail": "Дети",
}
out = pd.DataFrame(columns=["Class", "Title", "Description", "Text"])
for sheet, category in sheet_to_category.items():
    df = pd.read_excel("data.mail.xlsx", sheet)
    df["Class"] = category
    df["Text"] = df["Article Text"]
    df = df[["Class", "Title", "Description", "Text"]]
    out = pd.concat([out, df])
out.to_csv("data_processed.csv")
