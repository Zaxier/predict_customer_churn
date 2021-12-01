# Predict Customer Churn

## Project Description
Churn prediction model for bank data. Focus of the project is to develop fully tested, entirely an PEP8 compliant module for deployment.

## Running Files

#### 1. Set up Python environment
Install dependencies in the `environment.yml` file, ideally in a new conda environment by running 
```bash
conda env create -f environment.yml
```
This will create a new conda environment named `predict_churn`, unless you change the name of the environment by adjusting the first line of `environment.yml`


#### 2. Run churn_library.py
To process data, conduct eda, train models, and export model results run
```bash
python churn_library.py
```

#### Testing
With pytest installed you can simply run 
```bash
pytest
```
in the project directory to run the tests defined in `test_churn_script_logging_and_tests.py`.

Or you can run the test file directly, i.e.
```bash
python test_churn_script_logging_and_tests.py
```
