from typing import Optional
from fastapi import FastAPI
import mlflow
from mlflow.tracking import MlflowClient

import os
import pickle
import pandas as pd

relative_model_dev_path = '.'

model_name = "Bias Mitigation Telecom Churn"
mlflow.set_tracking_uri(f"sqlite:///{relative_model_dev_path}/Telecom_Churn_MLFlow.db")
runs = mlflow.search_runs(experiment_ids=['1'])
client = MlflowClient()

# sys.path.insert(1, relative_model_dev_path)
from auto_feat import MakeDataSet

md = MakeDataSet()

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predictions")
def predict(state:float, account_length:float, area_code:float, international_plan:float, voice_mail_plan:float, number_vmail_messages:float, total_day_minutes:float, total_day_calls:float, total_day_charge:float, total_eve_minutes:float, total_eve_calls:float, total_eve_charge:float, total_night_minutes:float, total_night_calls:float, total_night_charge:float, total_intl_minutes:float, total_intl_calls:float, total_intl_charge:float, customer_service_calls:float):
    
    data = None
    try:
        data = log_production_model()
    except Exception as e:
        print(e)
    
    if data == None:
        return {"predicted_class":"","error" : "No model in production in registry"}
    
    with open(f'{relative_model_dev_path}/{data[2:]}/model.pkl', "rb") as f:
        loaded_model = pickle.load(f)
    
    row_list = [
        state, 
        account_length, 
        area_code, 
        99999, 
        international_plan, 
        voice_mail_plan, 
        number_vmail_messages, 
        total_day_minutes, 
        total_day_calls, 
        total_day_charge, 
        total_eve_minutes, 
        total_eve_calls, 
        total_eve_charge, 
        total_night_minutes, 
        total_night_calls, 
        total_night_charge, 
        total_intl_minutes, 
        total_intl_calls, 
        total_intl_charge, 
        customer_service_calls, 
        999, 
    ]
    col_list = [
        'state',
        'account length',
        'area code',
        'phone number',
        'international plan',
        'voice mail plan',
        'number vmail messages',
        'total day minutes',
        'total day calls',
        'total day charge',
        'total eve minutes',
        'total eve calls',
        'total eve charge',
        'total night minutes',
        'total night calls',
        'total night charge',
        'total intl minutes',
        'total intl calls',
        'total intl charge',
        'customer service calls',
        'churn'
        ]

    row_df   = pd.DataFrame([row_list],columns = col_list)
   
    row_aif360 = md.decode_dataset(data_frame=row_df)
    
    prediction = loaded_model.predict(row_aif360.convert_to_dataframe()[0].drop('churn',axis=1))[0]
    
    return {"artifact_path": f'{data[2:]}/model.pkl', "predicted_class":int(prediction)}


def log_production_model():
    
    
    logged_model = None
    for reg_mod in client.list_registered_models():
        for versions in reg_mod.latest_versions:
            if versions.current_stage == 'Production':
                logged_model = versions.source
                break
    return logged_model