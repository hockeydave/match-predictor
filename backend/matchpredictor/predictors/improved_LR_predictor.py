from typing import List, Tuple, Optional

import pandas as pd

import numpy as np
from matchpredictor.matchresults.result import Fixture, Outcome, Result, Team
from matchpredictor.predictors.predictor import Predictor, Prediction
from numpy import float64
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder


class ImprovedLRPredictor(Predictor):
    def __init__(self, model: LogisticRegression, team_encoding: OneHotEncoder, pipe: Pipeline) -> None:
        self.pipe = pipe
        self.model = model
        self.team_encoding = team_encoding

    def predict(self, fixture: Fixture) -> Prediction:
        encoded_home_name = self.__encode_team(fixture.home_team)
        encoded_away_name = self.__encode_team(fixture.away_team)

        if encoded_home_name is None:
            return Prediction(outcome=Outcome.AWAY)
        if encoded_away_name is None:
            return Prediction(outcome=Outcome.HOME)
        team_features = [fixture.home_team.name, fixture.away_team.name, fixture.home_spi, fixture.away_spi,
                         fixture.home_imp, fixture.away_imp]

        df = pd.DataFrame(team_features).transpose()
        df.columns: List[str] = ['home_team', 'away_team', 'home_spi', 'away_spi', 'home_importance', 'away_importance']

        pred = self.pipe.predict(df)

        if pred > 0:
            return Prediction(outcome=Outcome.HOME)
        elif pred < 0:
            return Prediction(outcome=Outcome.AWAY)
        else:
            return Prediction(outcome=Outcome.DRAW)

    def __encode_team(self, team: Team) -> Optional[NDArray[float64]]:
        try:
            return self.team_encoding.transform(np.array(team.name).reshape(-1, 1))  # type: ignore
        except ValueError:
            return None


def build_model(results: List[Result]) -> Tuple[LogisticRegression, OneHotEncoder, Pipeline]:
    home_names = np.array([r.fixture.home_team.name for r in results])
    away_names = np.array([r.fixture.away_team.name for r in results])
    home_goals = np.array([r.home_goals for r in results])
    away_goals = np.array([r.away_goals for r in results])
    home_spis = np.array([r.fixture.home_spi for r in results])
    away_spis = np.array([r.fixture.away_spi for r in results])
    home_imps = np.array([r.fixture.home_imp for r in results])
    away_imps = np.array([r.fixture.away_imp for r in results])
    team_names = np.array(list(home_names) + list(away_names)).reshape(-1, 1)

    team_features = [home_names, away_names, home_spis, away_spis, home_imps, away_imps]

    df = pd.DataFrame(team_features).transpose()
    df.columns: List[str] = ['home_team', 'away_team', 'home_spi', 'away_spi', 'home_importance', 'away_importance']
    cat_columns = ["home_team", "away_team"]
    model = LogisticRegression(penalty="l2", fit_intercept=False, multi_class="ovr", C=1)
    team_encoding = OneHotEncoder(sparse=False).fit(team_names)
    pipe = make_pipeline(
        ColumnTransformer(
            transformers=[
                ("encode", team_encoding, cat_columns),
            ],
            remainder="passthrough"
        ),
        SimpleImputer(),
        model
    )

    y = np.sign(home_goals - away_goals)
    pipe: Pipeline = pipe.fit(df, y)
    return model, team_encoding, pipe


def train_improved_regression_predictor(results: List[Result]) -> Predictor:
    model, team_encoding, pipe = build_model(results)

    return ImprovedLRPredictor(model, team_encoding, pipe)
