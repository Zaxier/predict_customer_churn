# Predict Customer Churn

## Project Description
Churn prediction model for bank data. Focus of the project is to develop fully tested, entirely an PEP8 compliant module for deployment.

## Running Files

### 1. Set up Python environment
Install dependencies in the `environment.yml` file, ideally in a new conda environment by running 
```bash
$ conda env create -f environment.yml
```
This will create a new conda environment named `predict_churn`, unless you change the name of the environment by adjusting the first line of `environment.yml`


### 2. Run churn_library.py
To preprocess the raw data, conduct eda on that data, train the models, and export model results run the following from the command line:
```bash
$ python churn_library.py
```

### Testing - OPTIONAL
With pytest installed you can simply run `pytest` from the shell from within the project directory to run the tests defined in `test_churn_script_logging_and_tests.py`.

Alternatively you can run the test file directly, i.e.
```bash
$ python test_churn_script_logging_and_tests.py
```
