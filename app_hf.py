import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Настройка страницы
st.set_page_config(
    page_title="Предсказание оттока клиентов",
    page_icon="👋",
    layout="wide"
)

# Загрузка модели
try:
    with open('models/churn_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
except FileNotFoundError:
    st.error("Файл с моделью 'models/churn_pipeline.pkl' не найден. Убедитесь, что он загружен на Hugging Face Spaces.")
    st.stop()  # Останавливаем выполнение скрипта, если модели нет

# Словари для перевода
type_options = {
    'Ежемесячно': 'Month-to-month',
    'На 1 год': 'One year',
    'На 2 года': 'Two year'
}
payment_options = {
    'Электронный чек': 'Electronic check',
    'Чек по почте': 'Mailed check',
    'Банковский перевод (авто)': 'Bank transfer (automatic)',
    'Кредитная карта (авто)': 'Credit card (automatic)'
}
yes_no_options = { 'Да': 'Yes', 'Нет': 'No' }
not_connected_options = {
    'Да': 'Yes',
    'Нет': 'No',
    'Услуга не подключена': 'Not connected'
}

# Заголовок и описание
st.title('Прогнозирование оттока клиентов')
st.write("Это интерактивное демо использует модель CatBoost для предсказания вероятности ухода клиента. Заполните данные на панели слева и нажмите 'Предсказать'.")

# Боковая панель для ввода данных
st.sidebar.header('Параметры клиента')
inputs = {}

type_choice = st.sidebar.selectbox('Тип контракта:', list(type_options.keys()))
inputs['type'] = type_options[type_choice]

payment_choice = st.sidebar.selectbox('Метод оплаты:', list(payment_options.keys()))
inputs['payment_method'] = payment_options[payment_choice]

inputs['monthly_charges'] = st.sidebar.slider('Ежемесячный платеж:', min_value=18.0, max_value=120.0, value=70.0, step=0.1)
inputs['begin_date'] = st.sidebar.text_input('Дата начала контракта (ГГГГ-ММ-ДД):', value="2020-01-01")

col1, col2 = st.sidebar.columns(2)
with col1:
    billing_choice = st.sidebar.radio('Электронный счет:', list(yes_no_options.keys()))
    inputs['paperless_billing'] = yes_no_options[billing_choice]

    partner_choice = st.sidebar.radio('Наличие партнера:', list(yes_no_options.keys()))
    inputs['partner'] = yes_no_options[partner_choice]

    dependents_choice = st.sidebar.radio('Наличие иждивенцев:', list(yes_no_options.keys()))
    inputs['dependents'] = yes_no_options[dependents_choice]

    inputs['senior_citizen'] = st.sidebar.radio('Пенсионер:', [0, 1], format_func=lambda x: 'Да' if x == 1 else 'Нет')

with col2:
    lines_choice = st.sidebar.selectbox('Неск. тел. линий:', list(not_connected_options.keys()))
    inputs['multiple_lines'] = not_connected_options[lines_choice]

    security_choice = st.sidebar.selectbox('Онлайн-безопасность:', list(not_connected_options.keys()))
    inputs['online_security'] = not_connected_options[security_choice]

    backup_choice = st.sidebar.selectbox('Онлайн-резервирование:', list(not_connected_options.keys()))
    inputs['online_backup'] = not_connected_options[backup_choice]

    protection_choice = st.sidebar.selectbox('Защита устройства:', list(not_connected_options.keys()))
    inputs['device_protection'] = not_connected_options[protection_choice]

    support_choice = st.sidebar.selectbox('Тех. поддержка:', list(not_connected_options.keys()))
    inputs['tech_support'] = not_connected_options[support_choice]

    tv_choice = st.sidebar.selectbox('Стриминговое ТВ:', list(not_connected_options.keys()))
    inputs['streaming_tv'] = not_connected_options[tv_choice]

    movies_choice = st.sidebar.selectbox('Стриминг фильмов:', list(not_connected_options.keys()))
    inputs['streaming_movies'] = not_connected_options[movies_choice]

# Кнопка и логика предсказания 
if st.sidebar.button('🔮 Предсказать отток'):
    input_data = inputs.copy()

    try:
        snapshot_date = datetime.strptime('2020-02-01', '%Y-%m-%d')
        begin_date = datetime.strptime(input_data['begin_date'], '%Y-%m-%d')
        input_data['days_use'] = (snapshot_date - begin_date).days

        service_cols = ['tech_support', 'device_protection', 'online_backup', 'online_security']
        input_data['count_services'] = sum(1 for col in service_cols if input_data[col] == 'Yes')

        streaming_cols = ['streaming_movies', 'streaming_tv']
        input_data['count_streaming'] = sum(1 for col in streaming_cols if input_data[col] == 'Yes')

        features_df = pd.DataFrame([input_data])

        prediction = pipeline.predict(features_df)[0]
        probability = pipeline.predict_proba(features_df)[0][1]

        st.subheader('Результат прогноза:')
        if prediction == 1:
            st.error(f'Клиент скорее всего уйдет. Вероятность оттока: **{probability:.2%}**')
        else:
            st.success(f'Клиент скорее всего останется. Вероятность оттока: **{probability:.2%}**')

    except ValueError as e:
        st.error(f"Ошибка в формате данных! Проверьте, что дата введена в формате ГГГГ-ММ-ДД. Детали: {e}")

st.sidebar.info("Разработано для портфолио.")