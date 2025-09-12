from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from typing import Literal
from datetime import datetime

app = FastAPI()

# Загружаем обученный пайплайн
with open('models/churn_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)


# Определяем Pydantic модель для входных данных
class ClientFeatures(BaseModel):
    type: Literal['Month-to-month', 'One year', 'Two year']
    paperless_billing: Literal['Yes', 'No']
    payment_method: Literal['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    monthly_charges: float
    senior_citizen: int
    partner: Literal['Yes', 'No']
    dependents: Literal['Yes', 'No']
    multiple_lines: Literal['Yes', 'No', 'Not connected']

    begin_date: str = Field(..., example="2020-01-01") 

    online_security: Literal['Yes', 'No', 'Not connected']
    online_backup: Literal['Yes', 'No', 'Not connected']
    device_protection: Literal['Yes', 'No', 'Not connected']
    tech_support: Literal['Yes', 'No', 'Not connected']
    streaming_tv: Literal['Yes', 'No', 'Not connected']
    streaming_movies: Literal['Yes', 'No', 'Not connected']

    # Пример, как FastAPI будет показывать данные в документации
    class Config:
        json_schema_extra = {
            "example": {
                "type": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 70.7,
                "senior_citizen": 0,
                "partner": "No",
                "dependents": "No",
                "multiple_lines": "No",
                "begin_date": "2019-09-01",
                "online_security": "No",
                "online_backup": "No",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "No",
                "streaming_movies": "No"
            }
        }


def feature_engineering(input_data):
    snapshot_date = datetime.strptime('2020-02-01', '%Y-%m-%d')
    begin_date = datetime.strptime(input_data['begin_date'], '%Y-%m-%d')
    input_data['days_use'] = (snapshot_date - begin_date).days
    service_cols = ['tech_support', 'device_protection', 'online_backup', 'online_security']
    input_data['count_services'] = sum(1 for col in service_cols if input_data[col] == 'Yes')
    streaming_cols = ['streaming_movies', 'streaming_tv']
    input_data['count_streaming'] = sum(1 for col in streaming_cols if input_data[col] == 'Yes')
    return input_data


@app.get("/")
def read_root():
    return {"message": "Churn Prediction API is running!"}


@app.post("/predict/")
def predict(features: ClientFeatures):
    input_data = feature_engineering(features.model_dump())
    features_df = pd.DataFrame([input_data])

    prediction = pipeline.predict(features_df)[0]
    probability = pipeline.predict_proba(features_df)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability)
    }
