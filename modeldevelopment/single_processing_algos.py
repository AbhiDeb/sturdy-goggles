import logging

import numpy as np
from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.preprocessing.lfr import LFR

# Making modeldevelopment as a package and using it for importing functions and library
# import modeldevelopment.settings as settings

import settings
from engine import log_metrics
from load_data import save_dataframe
from metrics import compute_metrics
from visualize import plot_acc_spd_values

logger = logging.getLogger(__name__)


class SingleProcessingAlgos:
    def __init__(self, train, valid, test, exp_id):

        self.dataset_orig_train = train
        self.dataset_orig_test = test
        self.dataset_orig_valid = valid

        self.exp_id = exp_id
        self.model_hyper_tune = settings.NEED_TUNING

        self.bias_type = "After Tackling Bias"
        self.run_name = ''

        self.dataset_scale_train, self.dataset_scale_test, self.dataset_scale_valid = self.get_scaled_data()

        # ---------------------------------------------
        #               Pre-Processing
        # ---------------------------------------------

        self.process_type = "Pre-Processing"
        self.run_name = "Learning Fair Representations"
        self.lfr()

        # ---------------------------------------------
        #               In-Processing
        # ---------------------------------------------

        self.process_type = "In-Processing"
        self.run_name = "MetaFair Classifier"
        self.meta_fair_classifier()

        self.run_name = "PrejudiceRemover Classifier"
        self.prejudice_classifier()

    def get_scaled_data(self):
        try:
            # mas = settings.SCALAR()
            dataset_orig_train = self.dataset_orig_train.copy()
            dataset_orig_test = self.dataset_orig_test.copy(deepcopy=True)
            dataset_orig_valid = self.dataset_orig_valid.copy(deepcopy=True)

            # dataset_orig_train.features = mas.fit_transform(dataset_orig_train.features)
            # dataset_orig_test.features = mas.transform(dataset_orig_test.features)
            # dataset_orig_valid.features = mas.transform(dataset_orig_valid.features)

            return dataset_orig_train, dataset_orig_test, dataset_orig_valid

        except Exception as e:
            logger.error(e)

    """Learning Fair Representations"""

    def lfr(self):
        try:
            lfr_params = settings.LFR_PARAMS

            TR = LFR(**lfr_params)
            TR = TR.fit(self.dataset_scale_train, maxiter=5000, maxfun=5000)

            dataset_transfer_test = TR.transform(self.dataset_scale_test)
            dataset_transfer_valid = TR.transform(self.dataset_scale_valid)

            dataset_transf_test_new = self.dataset_scale_test.copy(deepcopy=True)
            dataset_transf_test_new.scores = dataset_transfer_test.scores

            dataset_transf_valid_new = self.dataset_scale_valid.copy(deepcopy=True)
            dataset_transf_valid_new.scores = dataset_transfer_valid.scores

            bal_acc_arr_transf = []
            disp_imp_arr_transf = []

            class_thresh_arr = np.linspace(0.01, 0.99, 100)

            for thresh in class_thresh_arr:
                fav_inds = dataset_transf_valid_new.scores > thresh
                dataset_transf_valid_new.labels[fav_inds] = 1.0
                dataset_transf_valid_new.labels[~fav_inds] = 0.0

                metric_test_aft = compute_metrics(self.dataset_scale_valid,
                                                  dataset_transf_valid_new,
                                                  False)

                bal_acc_arr_transf.append(metric_test_aft["Balanced accuracy"])
                disp_imp_arr_transf.append(metric_test_aft["Statistical parity difference"])

            threshold = class_thresh_arr[np.argmax(bal_acc_arr_transf)]

            fav_indices = dataset_transf_test_new.scores > threshold
            dataset_transf_test_new.labels[fav_indices] = dataset_transf_test_new.favorable_label
            dataset_transf_test_new.labels[~fav_indices] = dataset_transf_test_new.unfavorable_label

            lfr_metrics = compute_metrics(
                self.dataset_scale_test,
                dataset_transf_test_new,
                display=True)

            logger.info(f"Model : LFR({lfr_params})")

            log_metrics(run_name=self.run_name,
                        exp_id=self.exp_id,
                        classifier=TR,
                        classified_metric=lfr_metrics,
                        process_type=self.process_type,
                        bias_type=self.bias_type,
                        hyper_params=lfr_params)

            save_dataframe(dataset_transf_test_new.labels, self.run_name)

        except Exception as e:
            logger.error(e)

    """MetaFairClassifier"""

    def meta_fair_classifier(self):
        try:
            mfc_params = settings.META_FAIR_CLASSIFIER_PARAMS

            if self.model_hyper_tune:

                tau_values = settings.TAU_VALUES
                accuracy_values = list()
                spd_values = list()

                for each_tau in tau_values:
                    mfc_params["tau"] = each_tau
                    mfc_clf = MetaFairClassifier(**mfc_params).fit(
                        self.dataset_scale_train
                    )
                    mfc_cv_metrics = compute_metrics(
                        self.dataset_scale_valid,
                        mfc_clf.predict(self.dataset_scale_valid),
                        display=False,
                    )
                    accuracy_values.append(mfc_cv_metrics["Balanced accuracy"])
                    spd_values.append(mfc_cv_metrics["Statistical parity difference"])

                    # eta = eta_values[np.argmin(spd_values)]  */ Note :set the eta value for low SPD value. But it is
                    # completely depends upon the business objective./*

                plot_acc_spd_values(run_name=self.run_name,
                                    x_axis_threshold='TAU',
                                    threshold_values=tau_values,
                                    accuracy_values=accuracy_values,
                                    spd_values=spd_values)

                for max_index in np.array(accuracy_values).argsort()[-3:-1][::-1]:
                    mfc_params["tau"] = tau_values[max_index]
                    meta_clf = MetaFairClassifier(**mfc_params).fit(
                        self.dataset_scale_train
                    )
                    meta_clf_metrics = compute_metrics(
                        self.dataset_scale_test,
                        meta_clf.predict(self.dataset_scale_test),
                        display=False,
                    )

                    logger.info(f"Model  :MetaFairClassifier({mfc_params})\n")

                    log_metrics(run_name=self.run_name,
                                exp_id=self.exp_id,
                                classifier=meta_clf,
                                classified_metric=meta_clf_metrics,
                                process_type=self.process_type,
                                bias_type=self.bias_type,
                                hyper_params=mfc_params)

                mfc_params["tau"] = tau_values[int(np.argmax(accuracy_values))]

            meta_clf = MetaFairClassifier(**mfc_params).fit(
                self.dataset_scale_train
            )
            meta_clf_test = meta_clf.predict(self.dataset_scale_test)
            meta_classified_metrics = compute_metrics(
                self.dataset_scale_test,
                meta_clf_test,
                display=True,
            )

            logger.info(f"Model  :MetaFairClassifier({mfc_params})\n")

            log_metrics(run_name=self.run_name,
                        exp_id=self.exp_id,
                        classifier=meta_clf,
                        classified_metric=meta_classified_metrics,
                        process_type=self.process_type,
                        bias_type=self.bias_type,
                        hyper_params=mfc_params)

            save_dataframe(meta_clf_test.labels, self.run_name)

        except Exception as e:
            logger.error(e)

    """PrejudiceRemover """

    def prejudice_classifier(self):
        try:
            pjr_params = settings.PRE_JUDICE_REMOVER_PARAMS

            if self.model_hyper_tune:

                eta_values = settings.ETA_VALUES
                accuracy_values = list()
                spd_values = list()

                for each_eta in eta_values:
                    pjr_params["eta"] = each_eta
                    pjr_clf = PrejudiceRemover(**pjr_params).fit(
                        self.dataset_scale_train
                    )
                    pgr_cv_results = compute_metrics(
                        self.dataset_scale_valid,
                        pjr_clf.predict(self.dataset_scale_valid),
                        display=False,
                    )

                    accuracy_values.append(pgr_cv_results.get('Balanced accuracy'))
                    spd_values.append(pgr_cv_results.get("Statistical parity difference"))

                # eta = eta_values[np.argmin(spd_values)]  */ Note :set the eta value for low SPD value. But it is
                # completely depends upon the business objective./*

                plot_acc_spd_values(run_name=self.run_name,
                                    x_axis_threshold='ETA',
                                    threshold_values=eta_values,
                                    accuracy_values=accuracy_values,
                                    spd_values=spd_values)

                for max_index in np.array(accuracy_values).argsort()[-3:-1][::-1]:
                    pjr_params["eta"] = eta_values[max_index]
                    pjr_clf = PrejudiceRemover(**pjr_params).fit(
                        self.dataset_scale_train)
                    pjr_test_metrics = compute_metrics(
                        self.dataset_scale_test,
                        pjr_clf.predict(self.dataset_scale_test),
                        display=False,
                    )

                    logger.info(f"Model  :PrejudiceRemover({pjr_params})\n")

                    log_metrics(run_name=self.run_name,
                                exp_id=self.exp_id,
                                classifier=pjr_clf,
                                classified_metric=pjr_test_metrics,
                                process_type=self.process_type,
                                bias_type=self.bias_type,
                                hyper_params=pjr_params)

                pjr_params["eta"] = eta_values[int(np.argmax(accuracy_values))]

            pjr_clf = PrejudiceRemover(**pjr_params).fit(
                self.dataset_scale_train
            )
            pjr_clf_test = pjr_clf.predict(self.dataset_scale_test)
            prejudice_metrics = compute_metrics(
                self.dataset_scale_test,
                pjr_clf_test,
                display=True,
            )

            logger.info(f"Model  :PrejudiceRemover({pjr_params})\n")

            log_metrics(run_name=self.run_name,
                        exp_id=self.exp_id,
                        classifier=pjr_clf,
                        classified_metric=prejudice_metrics,
                        process_type=self.process_type,
                        bias_type=self.bias_type,
                        hyper_params=pjr_params)

            save_dataframe(pjr_clf_test.labels, self.run_name)

        except Exception as e:
            logger.error(e)
