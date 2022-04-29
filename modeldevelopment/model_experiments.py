import logging
import os
import traceback

import numpy as np
# In-Processing libraries
from aif360.algorithms.inprocessing import ExponentiatedGradientReduction
from aif360.algorithms.inprocessing import GridSearchReduction
# Post-Processing libraries
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing import RejectOptionClassification
from aif360.algorithms.preprocessing import DisparateImpactRemover
# Pre-Processing libraries
from aif360.algorithms.preprocessing import Reweighing
from sklearn.model_selection import GridSearchCV

# Making modeldevelopment as a package and using it for importing functions and library
# import modeldevelopment.settings as settings

import settings
from engine import log_metrics
from load_data import save_dataframe
from metrics import compute_metrics
from visualize import plot_acc_spd_values

# from load_data import save_dataframe

logger = logging.getLogger(__name__)


class ModelExperiment:
    """ Initializing and assigning the default values """

    def __init__(self,
                 classifier,
                 train,
                 valid,
                 test,
                 exp_id):

        self.classifier = classifier
        self.dataset_orig_train = train
        self.dataset_orig_valid = valid
        self.dataset_orig_test = test

        self.clf_name = type(classifier()).__name__
        self.exp_id = exp_id
        self.post_process_input_valid_data = None
        self.post_process_input_test_data = None
        self.model_hyper_tune = settings.NEED_TUNING

        self.best_class_thresh_val = {}
        self.run_name = ''

        self.dataset_scale_train, self.dataset_scale_test, self.dataset_scale_valid = self.get_scaled_data(

        )
        
        # ----------------------------------------------------------------
        #                           BaseModel
        # ----------------------------------------------------------------
        self.process_type = "Base Model"
        self.bias_type = "Without Tackling Bias"

        # Base Model
        self.run_name = self.clf_name + ' before Bias Mitigation'
        self.base_model, self.post_process_input_valid_data, self.post_process_input_test_data = self.run_base_model(
            self.dataset_orig_train)

        # -----------------------------------------------------------------
        #                         Pre-Processing
        # -----------------------------------------------------------------
        self.process_type = "Pre-Processing"
        self.bias_type = "After Tackling Bias"

        # Reweighing
        self.run_name = f"{self.clf_name} with Reweighing"
        _, _, _ = self.run_base_model(train_data=self.reweighing())

        # Disparate Impact Remover
        self.run_name = f"{self.clf_name} with Disparate ImpactRemover"
        self.disparate_impact_remover()

        # --------------------------------------------------------------
        #                        In-Processing
        # --------------------------------------------------------------
        self.process_type = "In-Processing"

        # Exponentiated Gradient Reduction
        self.run_name = f"{self.clf_name} with Exponentiated GradientReduction"
        self.expo_gradient_reduction()

        # GridSearch Reduction
        self.run_name = f"{self.clf_name} with GridSearch Reduction"
        self.grid_search_reduction()

        # -------------------------------------------------------------------
        #                         Post-Processing
        # -------------------------------------------------------------------
        self.process_type = "Post-Processing"

        # Calibrated EqOdds Postprocessing
        self.run_name = f"Calibrated EqOdds Postprocessing on {self.clf_name}"
        self.calibrated_eq_odds()

        # Reject Option Classification
        self.run_name = f"Reject Option Classification on {self.clf_name}"
        self.reject_option_classification()

        # EqOdds Postprocessing
        self.run_name = f"EqOddsPostprocessing on {self.clf_name}"
        self.eq_odds_post_processing()

    def get_scaled_data(self):
        try:
            scalar = settings.SCALAR()
            dataset_copy_train = self.dataset_orig_train.copy()
            dataset_copy_test = self.dataset_orig_test.copy(deepcopy=True)
            dataset_copy_valid = self.dataset_orig_valid.copy(deepcopy=True)

            # dataset_copy_train.features = scalar.fit_transform(
            #     dataset_copy_train.features)
            # dataset_copy_test.features = scalar.transform(
            #     dataset_copy_test.features)
            # dataset_copy_valid.features = scalar.transform(
            #     dataset_copy_valid.features)

            return dataset_copy_train, dataset_copy_test, dataset_copy_valid

        except Exception as e:
            logger.error(e)

    """ method to get the class threshold value from validation data"""

    def get_threshold_from_cv_data(self, dataset_copy_valid, process_type):
        try:
            num_thresh = 100
            ba_arr = []
            class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)

            for idx, class_thresh in enumerate(class_thresh_arr):
                fav_indices = dataset_copy_valid.scores > class_thresh
                dataset_copy_valid.labels[fav_indices] = dataset_copy_valid.favorable_label
                dataset_copy_valid.labels[~fav_indices] = dataset_copy_valid.unfavorable_label
                classified_metric = compute_metrics(
                    self.dataset_orig_valid,
                    dataset_copy_valid,
                    display=False
                )
                ba_arr.append(classified_metric["Balanced accuracy"])

            self.best_class_thresh_val["threshold"] = round(
                class_thresh_arr[np.argmax(ba_arr)], 3)

            logger.info(f"Best balanced accuracy for  {process_type} = %.4f" % round(
                np.max(ba_arr), 4))

        except Exception as e:
            print(traceback.format_exc())
            logger.error(e)

    """ Run Base Model with Grid Search"""

    def run_base_model(self, train_data):
        try:
            # scale = settings.SCALAR()

            # X_train = scale.fit_transform(train_data.features)
            X_train = train_data.features
            y_train = train_data.labels.ravel()
            w_train = train_data.instance_weights.ravel()

            if self.model_hyper_tune:

                param_grid = settings.HYPER_PARAMS.get(str(self.clf_name))
                grid_result = GridSearchCV(estimator=self.classifier(),
                                           param_grid=param_grid,
                                           cv=5,
                                           verbose=False,
                                           n_jobs=-1)
                grid_result.fit(X_train, y_train)
                clf = grid_result.best_estimator_
            else:
                clf = self.classifier()

            if self.process_type == "Pre-Processing":
                clf.fit(X_train, y_train, sample_weight=w_train)

            else:
                clf.fit(X_train, y_train)

            y_train_predictions = clf.predict(X_train)

            pos_ind = np.where(
                clf.classes_ == self.dataset_orig_train.favorable_label)[0][0]

            dataset_copy_train = self.dataset_orig_train.copy()
            dataset_copy_valid = self.dataset_orig_valid.copy(deepcopy=True)
            dataset_copy_test = self.dataset_orig_test.copy(deepcopy=True)

            # X_test = scale.transform(dataset_copy_test.features)
            # X_valid = scale.transform(dataset_copy_valid.features)

            X_test = dataset_copy_test.features
            X_valid = dataset_copy_valid.features

            dataset_copy_train.labels = y_train_predictions
            dataset_copy_valid.scores = clf.predict_proba(X_valid)[
                                        :, pos_ind
                                        ].reshape(-1, 1)
            dataset_copy_test.scores = clf.predict_proba(X_test)[
                                       :, pos_ind
                                       ].reshape(-1, 1)
            logger.info(f"Model : {clf}\n")

            if len(self.best_class_thresh_val) < 1:
                self.get_threshold_from_cv_data(dataset_copy_valid,
                                                self.process_type)
                logger.info(
                    f"Derived Class Threshold value = {self.best_class_thresh_val['threshold']} from validation data\n")

            fav_indices = dataset_copy_valid.scores > self.best_class_thresh_val["threshold"]
            dataset_copy_valid.labels[fav_indices] = dataset_copy_valid.favorable_label
            dataset_copy_valid.labels[~fav_indices] = dataset_copy_valid.unfavorable_label

            fav_indices = dataset_copy_test.scores > self.best_class_thresh_val["threshold"]
            dataset_copy_test.labels[fav_indices] = dataset_copy_test.favorable_label
            dataset_copy_test.labels[~fav_indices] = dataset_copy_test.unfavorable_label

            clf_test_metrics = compute_metrics(
                self.dataset_orig_test,
                dataset_copy_test,
                display=True,
            )

            log_metrics(run_name=self.run_name,
                        exp_id=self.exp_id,
                        classifier=clf,
                        classified_metric=clf_test_metrics,
                        process_type=self.process_type,
                        bias_type=self.bias_type,
                        hyper_params=clf.get_params())

            save_dataframe(dataset_copy_test.labels, self.run_name)

            return clf, dataset_copy_valid, dataset_copy_test

        except Exception as e:
            print(traceback.format_exc())
            logger.error(e)

    """Log Base Model result"""

    def mlflow_run(self, classifier, classified_metric, run_name):
        try:
            log_metrics(run_name=run_name,
                        exp_id=self.exp_id,
                        classifier=classifier,
                        classified_metric=classified_metric,
                        process_type=self.process_type,
                        bias_type=self.bias_type)
        except Exception as e:
            logging.error(e)

    """Re weighing"""

    def reweighing(self):
        try:
            rw_params = settings.REWEIGHING_PARAMS

            rw = Reweighing(**rw_params)
            rw.fit(self.dataset_orig_train)
            rw_trans_data = rw.transform(self.dataset_orig_train)

            logger.info(f"Model  : Reweighing({rw_params})\n")

            if not os.path.isfile(
                    f'./data/interim preprocessed data/Trained_data_after_reweighing_{settings.EXPERIMENT_NAME}.csv'):
                df = rw_trans_data.convert_to_dataframe()[0]
                df['sample weights'] = rw_trans_data.instance_weights.ravel()
                df.to_csv(
                    f'./data/interim preprocessed data/Trained_data_after_reweighing_{settings.EXPERIMENT_NAME}.csv',
                    index=False)

            return rw_trans_data

        except Exception as e:
            logger.error(e)

    """DisparateImpactRemover"""

    def disparate_impact_remover(self):
        try:
            dir_params = settings.DISPARATE_IMPACT_REMOVER_PARAMS
            column_index = self.dataset_scale_train.feature_names.index(
                settings.PRIVILEGED_ATTRIBUTE)

            valid_rep = self.dataset_scale_valid.copy(deepcopy=True)
            test_rep = self.dataset_scale_test.copy(deepcopy=True)

            if self.model_hyper_tune:

                repair_level_list = settings.REPAIR_LEVEL_LIST
                accuracy_values = []

                for each_level in repair_level_list:
                    dir_params["repair_level"] = each_level
                    di = DisparateImpactRemover(**dir_params)
                    train_rep = di.fit_transform(self.dataset_scale_train)

                    X_tr = np.delete(train_rep.features, column_index, axis=1)
                    X_cv = np.delete(valid_rep.features, column_index, axis=1)
                    y_tr = train_rep.labels.ravel()

                    clf = self.classifier()
                    clf.fit(X_tr, y_tr)

                    valid_copy_rep = valid_rep.copy()
                    valid_copy_rep.labels = clf.predict(X_cv)

                    cv_results = compute_metrics(valid_rep, valid_copy_rep,
                                                 display=False)
                    accuracy_values.append(cv_results["Balanced accuracy"])

                dir_params["repair_level"] = repair_level_list[np.argmax(
                    accuracy_values)]

            di_rem_clf = DisparateImpactRemover(**dir_params)
            train_rep = di_rem_clf.fit_transform(self.dataset_scale_train)

            X_tr = np.delete(train_rep.features, column_index, axis=1)
            X_te = np.delete(test_rep.features, column_index, axis=1)
            y_tr = train_rep.labels.ravel()

            clf = self.classifier()
            clf.fit(X_tr, y_tr)

            test_rep.labels = clf.predict(X_te)
            dir_metrics = compute_metrics(self.dataset_scale_test,
                                          test_rep,
                                          display=True)

            logger.info(f"Model  : DisparateImpactRemover({dir_params})\n")

            log_metrics(run_name=self.run_name,
                        exp_id=self.exp_id,
                        classifier=clf,
                        classified_metric=dir_metrics,
                        process_type=self.process_type,
                        bias_type=self.bias_type,
                        hyper_params=dir_params)

            save_dataframe(test_rep.labels, self.run_name)

        except Exception as e:
            logger.error(e)

    """ExponentiatedGradientReduction"""

    def expo_gradient_reduction(self):
        try:
            egr_params = settings.EXPONENTIATED_GRADIENT_REDUCTION_PARAMS
            egr_params["estimator"] = self.base_model

            if self.model_hyper_tune:

                accuracy_values = list()
                spd_values = list()
                eps_values = settings.EPS_VALUES

                for each_eps in eps_values:
                    egr_params["eps"] = each_eps
                    exp_grad_clf = ExponentiatedGradientReduction(**egr_params)
                    exp_grad_clf.fit(self.dataset_scale_train)

                    exp_grad_cv_metrics = compute_metrics(
                        self.dataset_scale_valid,
                        exp_grad_clf.predict(self.dataset_scale_valid),
                        display=False,
                    )

                    accuracy_values.append(
                        exp_grad_cv_metrics.get("Balanced accuracy"))
                    spd_values.append(
                        exp_grad_cv_metrics['Statistical parity difference'])

                plot_acc_spd_values(run_name=self.run_name,
                                    x_axis_threshold='EPS',
                                    threshold_values=eps_values,
                                    accuracy_values=accuracy_values,
                                    spd_values=spd_values)

                for max_index in np.array(accuracy_values).argsort()[-3:-1][::-1]:
                    egr_params["eps"] = eps_values[max_index]
                    exp_grad_clf = ExponentiatedGradientReduction(**egr_params)
                    exp_grad_clf.fit(self.dataset_scale_train)

                    exp_grad_test_metrics = compute_metrics(
                        self.dataset_scale_test,
                        exp_grad_clf.predict(self.dataset_scale_test),
                        display=False,
                    )

                    logger.info(
                        f"Model  :ExponentiatedGradientReduction({egr_params})\n")

                    log_metrics(run_name=self.run_name,
                                exp_id=self.exp_id,
                                classifier=exp_grad_clf,
                                classified_metric=exp_grad_test_metrics,
                                process_type=self.process_type,
                                bias_type=self.bias_type,
                                hyper_params=egr_params)

                egr_params["eps"] = eps_values[int(np.argmax(accuracy_values))]

            exp_grad_red = ExponentiatedGradientReduction(**egr_params)
            exp_grad_red.fit(self.dataset_scale_train)
            exp_grad_red_test = exp_grad_red.predict(self.dataset_scale_test)

            exp_grad_metrics = compute_metrics(
                self.dataset_scale_test,
                exp_grad_red_test,
                display=True,

            )
            logger.info(
                f"Model  :ExponentiatedGradientReduction({egr_params})\n")

            log_metrics(run_name=self.run_name,
                        exp_id=self.exp_id,
                        classifier=exp_grad_red,
                        classified_metric=exp_grad_metrics,
                        process_type=self.process_type,
                        bias_type=self.bias_type,
                        hyper_params=egr_params)

            save_dataframe(exp_grad_red_test.labels, self.run_name)

        except Exception as e:
            logger.error(e)

    """GridSearchReduction"""

    def grid_search_reduction(self):
        try:
            gsr_params = settings.GRID_SEARCH_REDUCTION_PARAMS
            gsr_params["estimator"] = self.base_model

            if self.model_hyper_tune:
                accuracy_values = list()
                spd_values = list()
                grid_values = settings.GRID_VALUES

                for each_grid in grid_values:
                    gsr_params["grid_size"] = each_grid

                    gsr_grid = GridSearchReduction(**gsr_params)
                    gsr_grid.fit(self.dataset_scale_train)

                    gsr_cv_metrics = compute_metrics(
                        self.dataset_scale_valid,
                        gsr_grid.predict(self.dataset_scale_valid),
                        display=False,
                    )
                    accuracy_values.append(
                        gsr_cv_metrics.get("Balanced accuracy"))
                    spd_values.append(gsr_cv_metrics.get(
                        "Statistical parity difference"))

                plot_acc_spd_values(run_name=self.run_name,
                                    x_axis_threshold='GRID SIZE',
                                    threshold_values=grid_values,
                                    accuracy_values=accuracy_values,
                                    spd_values=spd_values)

                for max_index in np.array(accuracy_values).argsort()[-3:-1][::-1]:
                    gsr_params["grid_size"] = grid_values[max_index]
                    gsr_clf = GridSearchReduction(**gsr_params)
                    gsr_clf.fit(self.dataset_scale_train)

                    gsr_test_metrics = compute_metrics(
                        self.dataset_scale_test,
                        gsr_clf.predict(self.dataset_scale_test),
                        display=False,
                    )

                    logger.info(
                        f"Model  : GridSearchReduction({gsr_params})\n")
                    log_metrics(run_name=self.run_name,
                                exp_id=self.exp_id,
                                classifier=gsr_clf,
                                classified_metric=gsr_test_metrics,
                                process_type=self.process_type,
                                bias_type=self.bias_type,
                                hyper_params=gsr_params)

                gsr_params["grid_size"] = grid_values[int(
                    np.argmax(accuracy_values))]

            gsr_clf = GridSearchReduction(**gsr_params)
            gsr_clf.fit(self.dataset_scale_train)
            gsr_clf_test = gsr_clf.predict(self.dataset_scale_test)

            gsr_metrics = compute_metrics(
                self.dataset_scale_test,
                gsr_clf_test,
                display=True,
            )

            logger.info(f"Model  : GridSearchReduction({gsr_params})\n")

            log_metrics(run_name=self.run_name,
                        exp_id=self.exp_id,
                        classifier=gsr_clf,
                        classified_metric=gsr_metrics,
                        process_type=self.process_type,
                        bias_type=self.bias_type,
                        hyper_params=gsr_params)

            save_dataframe(gsr_clf_test.labels, self.run_name)

        except Exception as e:
            logger.error(e)

    """CalibratedEqOddsPostprocessing"""

    def calibrated_eq_odds(self):
        try:
            ceo_params = settings.CALIBRATED_EQ_ODDS_POSTPROCESSING_PARAMS

            ceo_clf = CalibratedEqOddsPostprocessing(**ceo_params)
            ceo_clf = ceo_clf.fit(self.dataset_orig_valid,
                                  self.post_process_input_valid_data)
            ceo_clf_test = ceo_clf.predict(self.post_process_input_test_data)

            ceo_metrics = compute_metrics(
                self.dataset_orig_test,
                ceo_clf_test,
                display=True,
            )

            logger.info(
                f"Model  : CalibratedEqOddsPostprocessing({ceo_params})\n")
            log_metrics(run_name=self.run_name,
                        exp_id=self.exp_id,
                        classifier=ceo_clf,
                        classified_metric=ceo_metrics,
                        process_type=self.process_type,
                        bias_type=self.bias_type,
                        hyper_params=ceo_params)

            save_dataframe(ceo_clf_test.labels, self.run_name)

        except Exception as e:
            logger.error(e)

    """RejectOptionClassification"""

    def reject_option_classification(self):
        try:
            roc_params = settings.REJECT_OPTION_CLASSIFICATION_PARAMS

            roc_clf = RejectOptionClassification(**roc_params)
            roc_clf = roc_clf.fit(self.dataset_orig_valid,
                                  self.post_process_input_valid_data)

            roc_clf_test = roc_clf.predict(self.post_process_input_test_data)

            roc_metrics = compute_metrics(
                self.dataset_orig_test,
                roc_clf_test,
                display=True,
            )

            logger.info(f"Model  : RejectOptionClassification({roc_params})\n")
            logger.info("Optimal classification threshold  for ROC = %.4f" %
                        roc_clf.classification_threshold)
            logger.info("Optimal ROC margin = %.4f" % roc_clf.ROC_margin)

            log_metrics(run_name=self.run_name,
                        exp_id=self.exp_id,
                        classifier=roc_clf,
                        classified_metric=roc_metrics,
                        process_type=self.process_type,
                        bias_type=self.bias_type,
                        hyper_params=roc_params)

            save_dataframe(roc_clf_test.labels, self.run_name)

        except Exception as e:
            logger.error(e)

    """EqOddsPostprocessing"""

    def eq_odds_post_processing(self):
        try:
            eop_params = settings.EQ_ODDS_POST_PROCESSING_PARAMS

            eop_clf = EqOddsPostprocessing(**eop_params)
            eop_clf = eop_clf.fit(self.dataset_orig_valid,
                                  self.post_process_input_valid_data)

            eop_clf_test = eop_clf.predict(self.post_process_input_test_data)

            eop_metrics = compute_metrics(
                self.dataset_orig_test,
                eop_clf_test,
                display=True,
            )

            logger.info(f"Model  : EqOddsPostprocessing({eop_params}\n")
            log_metrics(run_name=self.run_name,
                        exp_id=self.exp_id,
                        classifier=eop_clf,
                        classified_metric=eop_metrics,
                        process_type=self.process_type,
                        bias_type=self.bias_type,
                        hyper_params=eop_params)

            save_dataframe(eop_clf_test.labels, self.run_name)

        except Exception as e:
            logger.error(e)
