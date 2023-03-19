from matchpredictor.matchresults.result import Fixture, Outcome
from matchpredictor.predictors.predictor import Prediction, Predictor


class AlphabeticPredictor(Predictor):

    def predict(self, fixture: Fixture) -> Prediction:
        if fixture.home_team.name > fixture.away_team.name:
            return Prediction(Outcome.HOME)
        elif fixture.home_team.name < fixture.away_team.name:
            return Prediction(Outcome.AWAY)
        else:
            return Prediction(Outcome.DRAW)
