import streamlit as st
import requests
import json

# Настройка страницы
st.set_page_config(
    page_title="Предсказание оттока клиентов",
    page_icon="👋",
    layout="wide"
)

# URL FastAPI-сервиса, работающего в Docker
API_URL = "http://localhost:8000/predict/"

st.title('Прогнозирование оттока клиентов')
st.write("""
Добро пожаловать в демонстрационное приложение для прогнозирования оттока клиентов.
Заполните параметры клиента на боковой панели, чтобы получить прогноз.
Это приложение отправляет запрос к ML-модели, развернутой с помощью FastAPI и Docker.
""")

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

st.sidebar.header('Параметры клиента')

inputs = {}


# Выбираем русский ключ
type_choice = st.sidebar.selectbox('Тип контракта', list(type_options.keys()))
inputs['type'] = type_options[type_choice]

payment_choice = st.sidebar.selectbox('Метод оплаты', list(payment_options.keys()))
inputs['payment_method'] = payment_options[payment_choice]

inputs['monthly_charges'] = st.sidebar.slider('Ежемесячный платеж', min_value=18.0, max_value=120.0, value=70.0, step=0.1)
inputs['begin_date'] = st.sidebar.text_input('Дата начала контракта', value="2020-01-01")

col1, col2 = st.sidebar.columns(2)
with col1:
    billing_choice = st.sidebar.radio('Электронный счет', list(yes_no_options.keys()))
    inputs['paperless_billing'] = yes_no_options[billing_choice]

    partner_choice = st.sidebar.radio('Наличие партнера', list(yes_no_options.keys()))
    inputs['partner'] = yes_no_options[partner_choice]

    dependents_choice = st.sidebar.radio('Наличие иждивенцев', list(yes_no_options.keys()))
    inputs['dependents'] = yes_no_options[dependents_choice]

    inputs['senior_citizen'] = st.sidebar.radio('Пенсионер', [0, 1], format_func=lambda x: 'Да' if x == 1 else 'Нет')

with col2:
    lines_choice = st.sidebar.selectbox('Неск. тел. линий', list(not_connected_options.keys()))
    inputs['multiple_lines'] = not_connected_options[lines_choice]

    security_choice = st.sidebar.selectbox('Онлайн-безопасность', list(not_connected_options.keys()))
    inputs['online_security'] = not_connected_options[security_choice]

    backup_choice = st.sidebar.selectbox('Онлайн-резервирование', list(not_connected_options.keys()))
    inputs['online_backup'] = not_connected_options[backup_choice]

    protection_choice = st.sidebar.selectbox('Защита устройства', list(not_connected_options.keys()))
    inputs['device_protection'] = not_connected_options[protection_choice]

    support_choice = st.sidebar.selectbox('Тех. поддержка', list(not_connected_options.keys()))
    inputs['tech_support'] = not_connected_options[support_choice]

    tv_choice = st.sidebar.selectbox('Стриминговое ТВ', list(not_connected_options.keys()))
    inputs['streaming_tv'] = not_connected_options[tv_choice]

    movies_choice = st.sidebar.selectbox('Стриминг фильмов', list(not_connected_options.keys()))
    inputs['streaming_movies'] = not_connected_options[movies_choice]



# Кнопка для запуска предсказания
if st.sidebar.button('🔮 Предсказать отток'):
    st.info("Отправка запроса к модели... Пожалуйста, подождите.")

    try:
        # Отправляем POST-запрос на наш FastAPI-сервис
        response = requests.post(API_URL, data=json.dumps(inputs))

        # Проверяем статус ответа
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            probability = result['churn_probability']

            #  Отображение результата
            st.subheader('Результат прогноза:')

            if prediction == 1:
                st.error(f'Клиент скорее всего уйдет. Вероятность оттока: **{probability:.2%}**')
            else:
                st.success(f'Клиент скорее всего останется. Вероятность оттока: **{probability:.2%}**')

            # Отображаем JSON, который был отправлен модели
            st.write("---")
            st.write("Данные, отправленные в модель (JSON):")
            st.json(inputs)

        else:
            st.error(f"Ошибка! Сервер вернул статус {response.status_code}.")
            st.error("Текст ошибки:")
            st.json(response.json())

    except requests.exceptions.ConnectionError:
        st.error("Не удалось подключиться к API. Убедитесь, что ваш Docker-контейнер с FastAPI запущен.")

st.sidebar.info("Это демо-приложение для демонстрации работы ML-модели, упакованной в Docker.")