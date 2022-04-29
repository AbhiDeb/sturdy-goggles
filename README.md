<h2 align="center">Binary Classification - Bias Mitigation w.r.t Dichotomous Features.<h2/>
This Framework help detect and mitigate bias in machine learning models throughout the AI application lifecycle.

```
├── biasdetection
│	├──notebooks
│	├──reports                         	
│	│   └── figures		                <- Generates analysis as HTML, PDF, LaTeX, etc.
│	└──bias_detection.py
├── modeldevelopment
│	├── data
│	│   ├── cleaned data                <- The final dataset after pre processing.
│	│   ├── interim preprocessed data   <- Intermediate data that has been transformed while execting alogs for e.g.,Reweighing - 
│	│ 	│ 	                       Before and After transforming training data.   
│	│   └── model results datasets      <- Each model predicted values attached to the
│	│   └── raw data                    <- The original, immutable data dump. 
│	├── logs                            <- Log details
│	├── mlruns                          <- This folder will be created while running the experiments.
│	├── notebooks                       <- Demo Jupyter notebooks for running and trouble shooting Bias Mitigation algorithms.
│	├── reports                         <- Generates analysis as HTML, PDF, LaTeX, etc.
│	│   └── figures                     <- Generates graphics and figures to be used in reporting.
│	└──app.py                           <- Starting file which runs data processing along with Pre,In and Post Processing algos.  
│	└──gen_log_table                    <- Returns a log details in tabular format│
│	└──engine.py                        <- Creates Mlflow experiments and logs all the details related to Bias Mitigation algos.
│	└──load_data.py                     <- To load the csv file from directory `./data/raw data` 
│	└──auto_feat.py                     <- Coverts the regular dataframe to AIF360 format dataframe and peforms data preprocessing steps. 
│	└──metrics.py                       <- Calculates the metrics like SPD,Balanced Accuracy ,etc.
│	└──model_experiments.py             <- Runs all the  Alogs for Bias Mitigation techniques.
│	├── README.md                       <- README for developers using this project.
│	├── requirements.txt                <- The requirements file for reproducing the analysis environment, e.g.
│	│                                      generated with `pip freeze > requirements.txt`
│	└──settings.py                      <- Configuartion file to set the parameters  for dichotomous protected features .
│										   (for any binary classification problem)
│	└──single_processing_alogs.py       <- Runs only once per one experiment, as it doesn't depend upon any non-bias mitigated ML algos. 
│	└──visualize.py                     <- Creates the plots under `reports/figures` when we perform hyper parameter tuning.
├── datasource                          <- The original, immutable data dump.

```

### Installation

```bash
    conda create --name <environment-name> python=3.7 -y
    activate <environment-name>
    pip install --user "aif360[ DisparateImpactRemover, Reductions, LFR, AdversarialDebiasing]"
    pip install --user mlflow 
    pip install --user xgboost
    pip install --user prettytable
    pip install --user plotly
    pip install -U kaleido --user
```

### Usage

```bash
  python app.py
  mlflow ui --backend-store-uri sqlite:///Heart_Disease_MLFlow.db
```

### Notes

```
1.The existing automated framework of Bias Mitigation doesn't explicitly require any database since mlflow
  itself stores all the experiment runs in a local folder (e.g. ./mlruns). However, if one wants to register a model , 
  then DB would be needed which can be executed by making just few changes in the code:
  
    a) pip install --user db-sqlite3
    b) In the engine.py script : uncomment line number -17
    d) Use the following command for running mlflow through terminal :
        mlflow ui --backend-store-uri "sqlite:///"name of the Database".db"
        example : mlflow ui --backend-store-uri "sqlite:///Telco_Customer_Churn.db"
        
2. Ideally, Test data set should be a good representative of Train data set , 
   which would then demonstrate the actual effectiveness of the Bias-Mitigation Algorithm.

```

#### Authors

```
Siddhant Khare, Ranjith Kumar and Abhitesh Debnath
```