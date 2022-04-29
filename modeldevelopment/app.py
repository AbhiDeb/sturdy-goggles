"""app.py"""
import logging
import time
import warnings as war

# Making modeldevelopment as a package and using it for importing functions and library
# import modeldevelopment.settings as settings

import settings
from engine import get_experiment_id
from load_data import load_dataframe
from auto_feat import MakeDataSet
from model_experiments import ModelExperiment
from single_processing_algos import SingleProcessingAlgos

war.filterwarnings(action="ignore")

artifact_loc = settings.ARTIFACT_LOCATION
experiment_name = settings.EXPERIMENT_NAME

if artifact_loc == "": artifact_loc = './mlruns'

log_fmt = "[%(asctime)s] - %(levelname)s -[%(filename)1s:%(lineno)d] %(message)s"
log_file = "./logs/{} at {}.log".format(str(experiment_name).lower(), time.strftime('%d-%m-%Y %H %M %S'))
# log_file = "./logs/{}_at_{}.log".format("_".join(str(experiment_name).split(" ")).lower(), time.strftime('%d-%m-%Y_%H%M%S'))

logging.basicConfig(filename=log_file,format=log_fmt,datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)

if __name__ == '__main__':
    try:
        logging.info("Loading Dataset...")
        md = MakeDataSet()
        # data_aif360 = md.decode_dataset(data_frame=load_dataframe())
        data_pandas_df = load_dataframe()
        # print(type(data_pandas_df))
        train_data, valid_data, test_data = md.train_valid_test_split(data_frame=data_pandas_df)

        exp_id = get_experiment_id()
        logging.info(f"All Experiments are stored under  {artifact_loc}  directory")
        logging.info(f"Model Experimentation for `{experiment_name}` is started...")

        MODELS = settings.MODEL_LIST
        for each_model in MODELS:
            print(f"Processing {each_model}")
            model_exp = ModelExperiment(classifier=each_model,
                                        train=train_data,
                                        valid=valid_data,
                                        test=test_data,
                                        exp_id=exp_id)

        sin_pro_algo = SingleProcessingAlgos(train=train_data,
                                             valid=valid_data,
                                             test=test_data,
                                             exp_id=exp_id)
    except Exception as e:
        logging.error(e)
