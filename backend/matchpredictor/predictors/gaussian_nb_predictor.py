from typing import List, Tuple, Optional

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from matchpredictor.matchresults.result import Fixture, Outcome, Result, Team
from matchpredictor.predictors.predictor import Predictor, Prediction


class GaussianNBPredictor(Predictor):
    def __init__(self, model: GaussianNB, team_encoding: OneHotEncoder, res_encoding: LabelEncoder) -> None:
        self.model = model
        self.team_encoding = team_encoding
        self.res_encoding = res_encoding

    def predict(self, fixture: Fixture) -> Prediction:
        encoded_home_name = self.__encode_team(fixture.home_team)
        encoded_away_name = self.__encode_team(fixture.away_team)

        if encoded_home_name is None:
            return Prediction(outcome=Outcome.AWAY)
        if encoded_away_name is None:
            return Prediction(outcome=Outcome.HOME)

        x: NDArray[float64] = np.concatenate([encoded_home_name, encoded_away_name], 1)  
        pred = self.model.predict(x)
       
        return Prediction(outcome=self.res_encoding.inverse_transform(pred)[0])

    def __encode_team(self, team: Team) -> Optional[NDArray[float64]]:
        try:
            return self.team_encoding.transform(np.array(team.name).reshape(-1, 1))  # type: ignore
        except ValueError:
            return None


def build_model(results: List[Result]) -> Tuple[GaussianNB, OneHotEncoder, LabelEncoder]:
    home_names = np.array([r.fixture.home_team.name for r in results])
    away_names = np.array([r.fixture.away_team.name for r in results])
    res = np.array([Outcome.convertToString(r.outcome) for r in results])

    team_names = np.array(list(home_names) + list(away_names)).reshape(-1, 1)
    
    team_encoding = OneHotEncoder(sparse=False).fit(team_names)
    y_encoding = LabelEncoder().fit(np.array(["home", "away", "draw"]))

    encoded_home_names = team_encoding.transform(home_names.reshape(-1, 1))
    encoded_away_names = team_encoding.transform(away_names.reshape(-1, 1))

    x: NDArray[float64] = np.concatenate([encoded_home_names, encoded_away_names], 1) 
    y = y_encoding.transform(res)
    model = GaussianNB()
    model.fit(x, y)

    return model, team_encoding, y_encoding


def train_nb_predictor(results: List[Result]) -> Predictor:
    model, team_encoding, res_encoding = build_model(results)

    return GaussianNBPredictor(model, team_encoding, res_encoding)
