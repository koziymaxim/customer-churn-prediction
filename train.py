import pandas as pd
import re
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostClassifier


def train_and_save_model():
    print("1. Загрузка данных...")

    # Загружаем данные по URL
    df_contract = pd.read_csv("https://code.s3.yandex.net/datasets/contract_new.csv")
    df_internet = pd.read_csv("https://code.s3.yandex.net/datasets/internet_new.csv")
    df_personal = pd.read_csv("https://code.s3.yandex.net/datasets/personal_new.csv")
    df_phone = pd.read_csv("https://code.s3.yandex.net/datasets/phone_new.csv")

    print("2. Предобработка и инжиниринг признаков")

    def camel_to_snake(name: str) -> str:
        name = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
        return name.lower()

    for df in [df_contract, df_internet, df_personal, df_phone]:
        df.columns = [camel_to_snake(c) for c in df.columns]
    print(df_contract.head())
    df_contract['total_charges'] = df_contract['total_charges'].replace(" ", 0).astype("float64")
    df_contract['begin_date'] = pd.to_datetime(df_contract['begin_date'], format='%Y-%m-%d')

    df_full = pd.merge(df_contract, df_internet, on='customer_id', how='outer')
    df_full = pd.merge(df_full, df_personal, on='customer_id', how='outer')
    df_full = pd.merge(df_full, df_phone, on='customer_id', how='outer')

    df_full.fillna('Not connected', inplace=True)

    df_full['terminated'] = (df_full['end_date'] != 'No').astype(int)

    end_date = pd.to_datetime(df_full['end_date'].replace('No', '2020-02-01'), format='%Y-%m-%d')
    df_full['days_use'] = (end_date - df_full['begin_date']).dt.days

    columns_to_sum_services = ['tech_support', 'device_protection', 'online_backup', 'online_security']
    columns_to_sum_streaming = ['streaming_movies', 'streaming_tv']

    df_full['count_services'] = (df_full[columns_to_sum_services] == 'Yes').sum(axis=1)
    df_full['count_streaming'] = (df_full[columns_to_sum_streaming] == 'Yes').sum(axis=1)

    columns_to_drop = [
        'end_date', 'begin_date', 'customer_id', 'gender', 'streaming_movies',
        'streaming_tv', 'tech_support', 'device_protection', 'online_backup',
        'online_security', 'internet_service', 'total_charges'
    ]
    df_full = df_full.drop(columns=columns_to_drop)

    print("3. Подготовка к обучению")

    X = df_full.drop('terminated', axis=1)
    y = df_full['terminated']

    ohe_columns = [
        'type', 'paperless_billing', 'payment_method', 
        'partner', 'dependents', 'multiple_lines'
    ]
    num_columns = [
        'monthly_charges', 'days_use', 'count_services', 'count_streaming'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), ohe_columns),
            ('num', StandardScaler(), num_columns)
        ],
        remainder='passthrough'
    )

    # Лучшие параметры для CatBoost
    best_params = {
        'n_estimators': 1600,
        'learning_rate': 0.05,
        'depth': 4,
        'l2_leaf_reg': 5,
        'random_strength': 5,
        'subsample': 0.8,
        'border_count': 128,
        'random_state': 20625,
        'verbose': 0
    }

    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', CatBoostClassifier(**best_params))
    ])

    print("4. Обучение модели")

    final_pipeline.fit(X, y)

    print("5. Сохранение пайплайна")
    path_to_save = 'models/churn_pipeline.pkl'

    with open(path_to_save, 'wb') as f:
        pickle.dump(final_pipeline, f)

    print(f"Пайплайн успешно сохранен в '{path_to_save}'")


if __name__ == '__main__':
    train_and_save_model()
