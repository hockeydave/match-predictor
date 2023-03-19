# Match predictor

The Match Predictor codebase contains an app with several predictors for football results.

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

To get your feet wet, create a predictive model that chooses the first team alphabetically to win.

Create a measurement test for your alphabet model in the backend/test/predictors directory. 
1.  Create measure_alphabet_predictor.py in test/predictor.  Measure for 33% accuracy, it's not going to be a great model.  
1.  Create your (alphabet_predictor.py) alphabet model in the backend/matchpredictor/predictors directory. Make sure that it inherits from Predictor.
1.  Make sure the tests pass by running make backend/measure.  You should see tests increase from 6 to 7.
1.  Add your alphabet model to ModelProvider in app.py and test_models_api.py
1.  Make sure your alphabet model shows up in the report by running make backend/report.
Develop a new model

## Now develop a more sophisticated model. Bonus points if you can get better than 50% accuracy predicting the 2021 English Premier League season.

Create a test for a new model, trained on past results (see a few of the other models for ideas in the backend/matchpredictor/predictors directory).
Start to implement your new model.
Add your new model to ModelProvider in app.py.
Using the measurement tests and the report, iterate on your model to improve the accuracy.