
KEYWORD_HINT_FEATURES = ['pix', 'uber', 'ifd', 'pag','pg', 'aplicação', 'salário', 'light']
DATE_FEATURES = ['Data']
NUMERIC_FEATURES = ['Valor']
TEXT_FEATURES = ['Descrição']
ALL_FEATURES = NUMERIC_FEATURES + KEYWORD_HINT_FEATURES + DATE_FEATURES + TEXT_FEATURES
TARGET_VARIABLE = ['categoria']


def add_keyword_hints(complete_dataset):
    for tag in KEYWORD_HINT_FEATURES:
        complete_dataset[tag] = complete_dataset['Descrição'].apply(lambda x: 1 if tag in x else 0)
    return complete_dataset[KEYWORD_HINT_FEATURES]


def add_date_features(complete_dataset):
    print(complete_dataset.head())
    complete_dataset['day'] = complete_dataset['Data'].dt.day
    complete_dataset['month'] = complete_dataset['Data'].dt.month
    complete_dataset.drop('Data', inplace=True, axis=1)
    return complete_dataset