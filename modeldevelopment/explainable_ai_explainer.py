import traceback
import pandas as pd
import numpy as np
import os
import pickle
import mlflow
from mlflow.tracking import MlflowClient
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

relative_model_dev_path = '.'

# import sys
# sys.path.insert(1, relative_model_dev_path)
# from modeldevelopment.load_data import load_dataframe
import settings
from auto_feat import MakeDataSet
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

def get_scaled_data(dataset_orig_train, num_columns):
    try:
        scalar = settings.SCALAR()
        dataset_copy_train = dataset_orig_train.copy()


        dataset_copy_train[num_columns] = scalar.fit_transform(
            dataset_copy_train[num_columns])

        return dataset_copy_train[num_columns]

    except Exception as e:
        print(traceback.format_exc())

def get_cat_num_columns(dataframe):
        """
        get categorical_names
        """
        ignore_columns = settings.COlUMN_TO_DROP + settings.Y_COLUMN
        cat_columns = []
        num_columns = []

        try:
            # if settings.COlUMN_TO_DROP:
            #     dataframe.drop(settings.COlUMN_TO_DROP,
            #                         axis=1, inplace=True)

            data = dataframe

            for col in data.columns:
                if (data[col].nunique() <= 51) and (col not in ignore_columns):
                    cat_columns.append(col)
                else:
                    if col not in ignore_columns:
                        num_columns.append(col)

            # logger.info(f"List of Numerical Columns : {self.num_columns}")
            # logger.info(f"List of Categorical Columns : {self.cat_columns}")
            return cat_columns, num_columns
        except Exception as e:
            print(traceback.format_exc())

def load_dataframe():
    """Load"""
    try:
        DATASET_PATH = "./data/model results datasets/Train_data.csv"
        data_frame = pd.read_csv(settings.DATASET_PATH)
        shuffled_df = data_frame.sample(
            frac=1, random_state=107).reset_index(drop=True)

        return shuffled_df
    except Exception as e:
        print(str(e))

if __name__ == '__main__':
    #     Load model
    loaded_model = None
    try:
        model_path = find_registered_model(name = "Bias Mitigation Telecom Churn", uri = f"sqlite:///{relative_model_dev_path}/Telecom_Churn_MLFlow.db")
        print(f'{relative_model_dev_path}/{model_path[2:]}/model.pkl')
        with open(f'{relative_model_dev_path}/{model_path[2:]}/model.pkl', "rb") as f:
            loaded_model = pickle.load(f)
        print(type(loaded_model))
    except Exception as e:
        print(traceback.format_exc())
        # print(e)
        
#     Get data
    df_ai360 = None
    try:
        df = load_dataframe()
        print(df.columns)
        cat_cols, num_cols = get_cat_num_columns(dataframe=df)
        
        scaler = settings.SCALAR()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        df_ai360 = md.decode_dataset(data_frame=df)

        train_data_scaled_df = df_ai360.convert_to_dataframe()[0]
        X_train = train_data_scaled_df.drop(settings.Y_COLUMN[0], axis = 1)
        y_train = train_data_scaled_df[settings.Y_COLUMN[0]]
        pass
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        
#         ExplainerDashboard
    try:
        
        explainer_new = ClassifierExplainer(loaded_model,
                                            X_train,
                                            y_train)
        ExplainerDashboard(explainer_new).run(port=9052)
        # Index 2124
    except Exception as e:
        print(e)
        print(traceback.format_exc())