import pandas as pd

if __name__ == '__main__':
    xls = pd.ExcelFile('../data/data.xlsx')
    df = pd.DataFrame()

    # Получаем имена всех листов в файле
    sheet_names = xls.sheet_names

    dict = {
        'kino.mail': 'Кино',
        'health.mail': 'Здоровье',
        'dom.mail': 'Дом',
        'deti.mail': 'Дети'
    }

    # Объединяем данные из всех листов в один DataFrame
    for sheet in sheet_names:
        tmp = xls.parse(sheet)
        tmp['Category'] = dict[sheet]
        tmp = tmp[['Article Text', 'Category']]
        df = pd.concat([df, tmp], ignore_index=True)

    df.to_csv('../data/data.csv', index=False)