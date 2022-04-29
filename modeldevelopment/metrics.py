import logging
from collections import OrderedDict
import traceback
import numpy as np

from aif360.metrics import ClassificationMetric

# Making modeldevelopment as a package and using it for importing functions and library
# import modeldevelopment.settings as settings

import settings

logger = logging.getLogger(__name__)


def compute_metrics(dataset_true, dataset_predictions, display=True):
    """ Compute the key metrics """
    try:
        classified_metric = ClassificationMetric(
            dataset_true,
            dataset_predictions,
            unprivileged_groups=settings.UNPRIVILEGED_GROUPS,
            privileged_groups=settings.PRIVILEGED_GROUPS,
        )

        # print("dataset_true")
        # df = dataset_true.convert_to_dataframe()[0]
        # df_pred = dataset_predictions.convert_to_dataframe()[0]
        # print(len(df[df['international plan'] == 0]))
        # print(len(df[df['international plan'] == 1]))
        # print("Target columns")
        # print(len(df[df[settings.Y_COLUMN[0]] == 0]))
        # print(len(df[df[settings.Y_COLUMN[0]] == 1]))
        # print("dataset_predictions")
        # df_pred = dataset_predictions.convert_to_dataframe()[0]
        # print(len(df_pred[df_pred['international plan'] == 0]))
        # print(len(df_pred[df_pred['international plan'] == 1]))
        # print("Target columns")
        # print(len(df_pred[df_pred[settings.Y_COLUMN[0]] == 0]))
        # print(len(df_pred[df_pred[settings.Y_COLUMN[0]] == 1]))

        metrics = OrderedDict()
        dataframe_predictions = dataset_predictions.convert_to_dataframe()[0]
        pred_0_values_count = len(dataframe_predictions[dataframe_predictions[settings.Y_COLUMN[0]] == 0])
        pred_1_values_count = len(dataframe_predictions[dataframe_predictions[settings.Y_COLUMN[0]] == 1])

        metrics["Balanced accuracy"] = 0.5 * (
                classified_metric.true_positive_rate() + classified_metric.true_negative_rate())
        metrics["Statistical parity difference"] = classified_metric.statistical_parity_difference()
        if pred_0_values_count != 0 and pred_1_values_count != 0:
            metrics["Disparate impact"] = classified_metric.disparate_impact()
        else:
            metrics["Disparate impact"] = np.nan
        metrics["error_rate_difference"] = classified_metric.error_rate_difference()
        metrics["error_rate_ratio"] = classified_metric.error_rate_ratio()
        metrics["average_odds_difference"] = classified_metric.average_odds_difference()
        metrics["equal_opportunity_difference"] = classified_metric.equal_opportunity_difference()

        if display:
            for k in metrics:
                print("%s  = %.4f" % (k, metrics[k]))
            print('\n')
            
        return metrics
    except Exception as e:
        print(traceback.format_exc())
        logger.error(e)
        logger.error(traceback.format_exc())
