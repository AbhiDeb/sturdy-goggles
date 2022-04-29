import logging

import mlflow
from mlflow.tracking import MlflowClient

# Making modeldevelopment as a package and using it for importing functions and library
# import modeldevelopment.settings as settings

import settings
from gen_log_table import gen_table

logger = logging.getLogger(__name__)

artifact_loc = settings.ARTIFACT_LOCATION
experiment_name = settings.EXPERIMENT_NAME
registry_uri = settings.BACKEND_STORE


def get_experiment_id():
    logging.info("Staring ML Flow...")
#     mlflow.set_registry_uri(registry_uri)
    mlflow.set_tracking_uri(registry_uri)
    try:
        exp_id = mlflow.create_experiment(experiment_name)
    except Exception as e:
        client = MlflowClient()
        exp_id = client.get_experiment_by_name(experiment_name).experiment_id
        logging.error(e)

    mlflow.set_experiment(experiment_name)

    return exp_id


def log_metrics(run_name, exp_id, classifier, classified_metric, process_type, bias_type, hyper_params=None):
    try:
        # pass
        with mlflow.start_run(run_name=run_name, experiment_id=exp_id):

            hyper_params_string = ''
            hyper_params_json = {}

            if hyper_params:
                for key, val in hyper_params.items():
                    hyper_params_string = hyper_params_string + \
                                          str(key) + '=' + str(val) + ','
                    hyper_params_json[str(key)] = str(val)

            for key, value in classified_metric.items():
                mlflow.log_metric(str(key), value)

            logging.info(
                f"RESULTS :\n\n{gen_table(run_name, classified_metric, process_type, bias_type)}\n")

            mlflow.log_param("Process Type", process_type)
            mlflow.log_param("Bias Type", bias_type)
            mlflow.log_param("Hyper Parameters", hyper_params_string)
            mlflow.sklearn.log_model(classifier, "Models_Storage")
            mlflow.log_dict(dictionary=hyper_params_json,
                            artifact_file=f"{run_name}.json")

    except Exception as e:
        logger.error(e)
