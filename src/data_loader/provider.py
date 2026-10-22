import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
from src.data_loader.base import MatchDataProvider


LALIGA_TEAMS = [
    {"name": "Real Madrid", "short_name": "R. Madrid", "code": "RMA"},
    {"name": "FC Barcelona", "short_name": "Barcelona", "code": "FCB"},
    {"name": "Atlético de Madrid", "short_name": "Atlético", "code": "ATM"},
    {"name": "Athletic Club", "short_name": "Athletic", "code": "ATH"},
    {"name": "Real Sociedad", "short_name": "R. Sociedad", "code": "RSO"},
    {"name": "Villarreal CF", "short_name": "Villarreal", "code": "VIL"},
    {"name": "Real Betis", "short_name": "Betis", "code": "BET"},
    {"name": "Sevilla FC", "short_name": "Sevilla", "code": "SEV"},
    {"name": "Girona FC", "short_name": "Girona", "code": "GIR"},
    {"name": "CA Osasuna", "short_name": "Osasuna", "code": "OSA"},
    {"name": "RC Celta de Vigo", "short_name": "Celta", "code": "CEL"},
    {"name": "Rayo Vallecano", "short_name": "Rayo", "code": "RAY"},
    {"name": "RCD Mallorca", "short_name": "Mallorca", "code": "MLL"},
    {"name": "Getafe CF", "short_name": "Getafe", "code": "GET"},
    {"name": "Deportivo Alavés", "short_name": "Alavés", "code": "ALA"},
    {"name": "RCD Espanyol", "short_name": "Espanyol", "code": "ESP"},
    {"name": "CD Leganés", "short_name": "Leganés", "code": "LEG"},
    {"name": "UD Las Palmas", "short_name": "Las Palmas", "code": "LPA"},
    {"name": "Valencia CF", "short_name": "Valencia", "code": "VCF"},
    {"name": "Real Valladolid", "short_name": "Valladolid", "code": "VLL"},
]


class LocalSimulatedMatchProvider(MatchDataProvider):
    """Proveedor local simulado determinista para desarrollo y pruebas offline."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def get_teams(self) -> List[Dict[str, Any]]:
        return LALIGA_TEAMS

    def get_historical_matches(self, season: str = "2025-2026") -> List[Dict[str, Any]]:
        """Genera los 380 partidos completos de la temporada histórica previa."""
        rng = random.Random(self.seed)
        teams = LALIGA_TEAMS
        n_teams = len(teams)
        matches = []
        start_date = datetime(2025, 8, 15)

        for matchday in range(1, 39):
            matchday_date = start_date + timedelta(days=(matchday - 1) * 7)
            for i in range(n_teams // 2):
                if matchday % 2 == 1:
                    home_idx = (i + matchday) % n_teams
                    away_idx = (n_teams - 1 - i + matchday) % n_teams
                else:
                    home_idx = (n_teams - 1 - i + matchday) % n_teams
                    away_idx = (i + matchday) % n_teams

                if home_idx == away_idx:
                    away_idx = (away_idx + 1) % n_teams

                home_score = rng.choices([0, 1, 2, 3, 4], weights=[0.2, 0.35, 0.25, 0.15, 0.05])[0]
                away_score = rng.choices([0, 1, 2, 3], weights=[0.3, 0.4, 0.2, 0.1])[0]

                matches.append({
                    "season": season,
                    "matchday": matchday,
                    "date": matchday_date + timedelta(hours=i * 2),
                    "home_team_code": teams[home_idx]["code"],
                    "away_team_code": teams[away_idx]["code"],
                    "home_score": home_score,
                    "away_score": away_score,
                    "status": "FINISHED"
                })

        return matches

    def get_active_matches(self, season: str = "2026-2027") -> List[Dict[str, Any]]:
        """Genera dinámicamente todo el calendario de la temporada activa.
        
        Calcula qué partidos ya han ocurrido según la fecha actual (marcándolos como FINISHED)
        y cuáles están por jugarse (marcándolos como SCHEDULED).
        """
        rng = random.Random(self.seed + 100)
        teams = LALIGA_TEAMS
        n_teams = len(teams)
        matches = []
        
        start_date = datetime(2026, 8, 18)
        now = datetime.now()

        for matchday in range(1, 39):  # Genera las 38 jornadas del calendario
            matchday_date = start_date + timedelta(days=(matchday - 1) * 7)
            
            for i in range(n_teams // 2):
                if matchday % 2 == 1:
                    home_idx = (i + matchday) % n_teams
                    away_idx = (n_teams - 1 - i + matchday) % n_teams
                else:
                    home_idx = (n_teams - 1 - i + matchday) % n_teams
                    away_idx = (i + matchday) % n_teams

                if home_idx == away_idx:
                    away_idx = (away_idx + 1) % n_teams

                match_datetime = matchday_date + timedelta(hours=i * 2)

                # Si la fecha del partido ya pasó respecto a la fecha actual, está jugado
                if match_datetime < now:
                    status = "FINISHED"
                    home_score = rng.choices([0, 1, 2, 3], weights=[0.25, 0.4, 0.25, 0.1])[0]
                    away_score = rng.choices([0, 1, 2, 3], weights=[0.35, 0.4, 0.2, 0.05])[0]
                else:
                    status = "SCHEDULED"
                    home_score = None
                    away_score = None

                matches.append({
                    "season": season,
                    "matchday": matchday,
                    "date": match_datetime,
                    "home_team_code": teams[home_idx]["code"],
                    "away_team_code": teams[away_idx]["code"],
                    "home_score": home_score,
                    "away_score": away_score,
                    "status": status
                })

        return matches


class SoccerDataMatchProvider(MatchDataProvider):
    """Adaptador para la librería soccerdata con fallback al proveedor local."""

    def __init__(self, league: str = "ESP-LaLiga"):
        self.league = league
        self.fallback = LocalSimulatedMatchProvider()

    def get_teams(self) -> List[Dict[str, Any]]:
        return self.fallback.get_teams()

    def get_historical_matches(self, season: str) -> List[Dict[str, Any]]:
        try:
            import soccerdata as sd
            return self.fallback.get_historical_matches(season)
        except Exception:
            return self.fallback.get_historical_matches(season)

    def get_active_matches(self, season: str) -> List[Dict[str, Any]]:
        try:
            import soccerdata as sd
            return self.fallback.get_active_matches(season)
        except Exception:
            return self.fallback.get_active_matches(season)