# Match predictor

The Match Predictor codebase contains an app with several predictors for football results.  
It's reading football match data from 531.org
While this is a football match predictor it's also a framework for automated testing of models
with code changes over time.

## Local development

Follow the instructions below to get the app up and running on your machine.

1.  Install Python 3.11 and a recent version of NPM.
1.  Install dependencies and run tests.
    ```shell
    make install test
    ```
1.  View the list of available tasks
    ```shell
    make
    ```

## Backend

Here are a few tasks that are useful when running the backend app.
Make sure they all run on your machine.

1.  Run tests
    ```shell
    make backend/test

1.  Run model measurement tests (trains model)
    ```shell
    make backend/measure
    ```

1.  Run server
    ```shell
    make backend/run
    ```

1.  Run an accuracy report (runs models and generates accuracy reports)
    ```shell
    make backend/report
    ```

## Frontend

Here are a few tasks that are useful when running the frontend app.
Make sure they all run on your machine.

1.  Run tests
    ```shell
    make frontend/test
    ```

1.  Run server
    ```shell
    make frontend/run
    ```

## Integration tests

If it's helpful, you may want to run integration tests during development.
Do so with the tasks below.

1.  Run tests
    ```shell
    make integration/test
    ```

1.  Interactive mode
    ```shell
    make integration/run
    ```
## Actual Code Assignment after above tests run properly

## Step-By-Step Assignment Instructions
less
## Sample model

Some models created are for validating the framework with automated testing (e.g. alphabetic/home team) models.  Otherw are intended
to be actual models to see how they perform (e.g. LR modell, improved LR model, )

Create a measurement test for your alphabet model in the backend/test/predictors directory. 
1.  Measure_alphabet_predictor.py in test/predictor.  Measure for 33% accuracy, it's not going to be a great model.  
1.  Created  (alphabet_predictor.py) alphabet model in the backend/matchpredictor/predictors directory. Make sure that it inherits from Predictor.
1.  Make sure the tests pass by running make backend/measure.  There are 7 model tests.
1.  Added  alphabet model to ModelProvider in app.py and test_models_api.py
1.  Make sure added alphabet model shows up in the report by running make backend/report.
Automated testing of models:  adaboost, alphabetic (just for testing framework), gaussian NB, home team (baseline), Improved LR (more featurs, Linear regression,
2. and past results predictor.

## If you want to develop a more sophisticated model. Bonus points if you can get better than 50% accuracy predicting the 2021 English Premier League season.

Create a test for a new model, trained on past results (see a few of the other models for ideas in the backend/matchpredictor/predictors directory).
Start to implement your new model.
Add your new model to ModelProvider in app.py.
Using the measurement tests and the report, iterate on your model to improve the accuracy.