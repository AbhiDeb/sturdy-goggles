import traceback
import pandas as pd
import numpy as np
import os
import pickle
import mlflow
from mlflow.tracking import MlflowClient

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab, ClassificationPerformanceTab
from evidently.pipeline.column_mapping import ColumnMapping


# relative_model_dev_path = '../modeldevelopment/'
relative_model_dev_path = '.'

# import sys
# sys.path.insert(1, relative_model_dev_path)
# from modeldevelopment.auto_feat import MakeDataSet
from auto_feat import MakeDataSet
import settings
md = MakeDataSet()

def find_registered_model(name, uri):
    model_name = name
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_ids=['1'])

    client = MlflowClient()
    logged_model = None
    for reg_mod in client.list_registered_models():
        for versions in reg_mod.latest_versions:
            if versions.current_stage == 'Production':
                logged_model = versions.source
                return logged_model

def get_scaled_data(dataset_orig_train,dataset_orig_test):
    try:
        scalar = settings.SCALAR()
        dataset_copy_train = dataset_orig_train.copy()
        dataset_copy_test = dataset_orig_test.copy(deepcopy=True)

        dataset_copy_train.features = scalar.fit_transform(
            dataset_copy_train.features)
        dataset_copy_test.features = scalar.transform(
            dataset_copy_test.features)

        return dataset_copy_train, dataset_copy_test

    except Exception as e:
        print(traceback.format_exc())

def train_valid_test_split(data_frame):
    """
    dataframe
    """
    try:
        dataset_orig_train, dataset_orig_test = data_frame.split(
            [0.7], shuffle=False)

        return dataset_orig_train, dataset_orig_test

    except Exception as e:
        print(traceback.format_exc())

def load_dataframe():
    """Load"""
    try:
        data_frame = pd.read_csv(settings.DATASET_PATH)
        shuffled_df = data_frame.sample(
            frac=1, random_state=42).reset_index(drop=True)

        return shuffled_df
    except Exception as e:
        print(traceback.format_exc())
        
if __name__ == '__main__':
    #     Load model
    loaded_model = None
    X_train = None
    X_test = None

    try:
        model_path = find_registered_model(name = "Bias Mitigation Telecom Churn", uri = f"sqlite:///{relative_model_dev_path}/Telecom_Churn_MLFlow.db")
        print(f'{relative_model_dev_path}/{model_path[2:]}/model.pkl')
        with open(f'{relative_model_dev_path}/{model_path[2:]}/model.pkl', "rb") as f:
            loaded_model = pickle.load(f)
        print(type(loaded_model))
        
    except Exception as e:
        print(traceback.format_exc())
        
    try:
        df = load_dataframe()

        # df_ai360 = md.decode_dataset(data_frame=df)
        train_data, val_data, test_data = md.train_valid_test_split(data_frame=df)
        X_train = train_data.features
        X_test = test_data.features
        
        pass
    except Exception as e:
        print(traceback.format_exc())
        
#         Model_Monitoring
    try:
        # print("11")
        ref_prediction  = loaded_model.predict(X_train)
        prod_prediction = loaded_model.predict(X_test)

        reference  = train_data.convert_to_dataframe()[0]
        production = test_data.convert_to_dataframe()[0]


        reference['prediction'] = ref_prediction
        production['prediction'] = prod_prediction

        cat_columns = list(set(md.cat_columns))
        num_columns = list(set(md.num_columns))
        
        column_mapping = ColumnMapping(
            target = 'churn',
            prediction = 'prediction',
            categorical_features = cat_columns,
            numerical_features= num_columns
        )


        classification_perfomance_dashboard = Dashboard(tabs=[ClassificationPerformanceTab() ,CatTargetDriftTab() ,DataDriftTab()])

        classification_perfomance_dashboard.calculate(reference,production,column_mapping)

        classification_perfomance_dashboard.save('./monitoring_reports/Telecom_Churn_Model_Monitoring.html')

        pass
    except Exception as e:
        print(traceback.format_exc())