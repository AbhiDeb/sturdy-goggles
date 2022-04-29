import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATASET_PATH = r"../datasource/churn.csv"
COlUMN_TO_DROP = ['phone number']
Y_COLUMN = ["churn"]

PRIVILEGED_ATTRIBUTE = "international plan"
PRIVILEGED_GROUPS = [{"international plan": 0}]
UNPRIVILEGED_GROUPS = [{"international plan": 1}]
FAVORABLE_CLASS = [0]

EXPERIMENT_NAME = "Bias Mitigation Telecom Churn"
BACKEND_STORE = "sqlite:///Telecom_Churn_MLFlow.db"
# BACKEND_STORE = "http://localhost:1234"
ARTIFACT_LOCATION = ""
NEED_TUNING = True
SCALAR = StandardScaler
AUTO_FEATURES = False

MODEL_LIST = [
    LogisticRegression,
    RandomForestClassifier,
    XGBClassifier
]

HYPER_PARAMS = \
    {
        'LogisticRegression':
            [
                {
                    'C': [10 ** i for i in range(-5, 10)]
                }
            ],
        "RandomForestClassifier":
            [
                {
                    'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
                    'max_depth': [int(x) for x in np.linspace(2, 8, num=2)]
                }
            ],
        "XGBClassifier":
            [

                {
                    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                    "max_depth": [3, 4, 5, 10, 12, 15],
                    "min_child_weight": [1, 3, 5, 7],
                }

            ]
    }

# ----------PreProcessing----------
REWEIGHING_PARAMS = \
    {
        'unprivileged_groups': UNPRIVILEGED_GROUPS,
        'privileged_groups': PRIVILEGED_GROUPS
    }

DISPARATE_IMPACT_REMOVER_PARAMS = \
    {
        "sensitive_attribute": PRIVILEGED_ATTRIBUTE,
        "repair_level": 1.0
    }
REPAIR_LEVEL_LIST = [round(i, 2) for i in np.linspace(0., 1., 11)]

LFR_PARAMS = \
    {
        "unprivileged_groups": UNPRIVILEGED_GROUPS,
        "privileged_groups": PRIVILEGED_GROUPS,
        "k": 10,
        "Ax": 0.1,
        "Ay": 1.0,
        "Az": 2.0,
        "verbose": 1
    }

# ----------InProcessing----------
EXPONENTIATED_GRADIENT_REDUCTION_PARAMS = \
    {
        "eps": 0.01,
        "T": 50,
        "constraints": "EqualizedOdds",
        "drop_prot_attr": False
    }
EPS_VALUES = [round(i, 2) for i in np.linspace(0.01, 0.1, 5)]

GRID_SEARCH_REDUCTION_PARAMS = \
    {
        "constraints": "EqualizedOdds",
        "drop_prot_attr": False,
        "grid_size": 10,
        "prot_attr": PRIVILEGED_ATTRIBUTE,
        "constraint_weight": 0.5
    }
GRID_VALUES = [int(i) for i in np.linspace(10, 30, 5)]

META_FAIR_CLASSIFIER_PARAMS = \
    {
        "sensitive_attr": PRIVILEGED_ATTRIBUTE,
        "type": "fdr",
        "tau": 0.8
    }
TAU_VALUES = [round(i, 1) for i in np.linspace(0.0, 1.0, 10)]

PRE_JUDICE_REMOVER_PARAMS = \
    {
        "sensitive_attr": PRIVILEGED_ATTRIBUTE,
        "eta": 1.0
    }
ETA_VALUES = [round(i, 1) for i in np.linspace(0.0, 1.0, 5)]

# ---------- Post Processing----------
CALIBRATED_EQ_ODDS_POSTPROCESSING_PARAMS = \
    {
        "cost_constraint": "weighted",
        'unprivileged_groups': UNPRIVILEGED_GROUPS,
        'privileged_groups': PRIVILEGED_GROUPS
    }

REJECT_OPTION_CLASSIFICATION_PARAMS = \
    {
        'unprivileged_groups': UNPRIVILEGED_GROUPS,
        'privileged_groups': PRIVILEGED_GROUPS,
        'low_class_thresh': 0.01,
        'high_class_thresh': 0.99,
        'num_class_thresh': 100,
        'num_ROC_margin': 50,
        'metric_name': "Statistical parity difference",
        'metric_ub': 0.05,
        'metric_lb': -0.05
    }

EQ_ODDS_POST_PROCESSING_PARAMS = \
    {
        "unprivileged_groups": UNPRIVILEGED_GROUPS,
        "privileged_groups": PRIVILEGED_GROUPS,
    }
