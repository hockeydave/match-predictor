import csv
from typing import Dict, Callable, List, Optional, cast

import numpy as np
import requests

from matchpredictor.matchresults.result import Result, Fixture, Team, Outcome


def training_results(
        csv_location: str,
        year: int,
        result_filter: Callable[[Result], bool] = lambda result: True,
) -> List[Result]:
    return load_results(csv_location, lambda r: result_filter(r) and r.season < year)


def validation_results(
        csv_location: str,
        year: int,
        result_filter: Callable[[Result], bool] = lambda result: True
) -> List[Result]:
    return load_results(csv_location, lambda r: result_filter(r) and r.season == year)


def load_results(
        csv_location: str,
        result_filter: Callable[[Result], bool] = lambda result: True,
) -> List[Result]:
    def match_outcome(home_goals: int, away_goals: int) -> Outcome:
        if home_goals > away_goals:
            return Outcome.HOME
        if away_goals > home_goals:
            return Outcome.AWAY
        return Outcome.DRAW

    def float_try(strf: str) -> float:
        try:
            return float(strf)
        except ValueError:
            return 0.0

    def result_from_row(row: Dict[str, str]) -> Optional[Result]:
        try:
            home_goals = int(row['score1'])
            away_goals = int(row['score2'])

            home_spi = float_try(row['spi1'])
            away_spi = float_try(row['spi2'])
            home_imp = float_try(row['importance1'])
            away_imp = float_try(row['importance2'])

            return Result(
                fixture=Fixture(
                    home_team=Team(row['team1']),
                    away_team=Team(row['team2']),
                    league=row['league'],
                    home_spi=home_spi,
                    away_spi=away_spi,
                    home_imp=home_imp,
                    away_imp=away_imp
                ),
                outcome=match_outcome(home_goals, away_goals),
                home_goals=home_goals,
                away_goals=away_goals,
                season=int(row['season'])
            )
        except (KeyError, ValueError):
            return None

    training_data = requests.get(csv_location).text

    rows = csv.DictReader(training_data.splitlines())
    results = filter(lambda r: type(r) is Result and result_filter(r), map(result_from_row, rows))

    return cast(List[Result], list(results))
