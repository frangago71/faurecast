import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional


class FeatureBuilder:
    """Calculador dinámico con métricas separadas por condición Local/Visitante."""

    def __init__(
        self,
        initial_elo: float = 1500.0,
        initial_sigma: float = 200.0,
        k_factor: float = 32.0,
        ewma_alpha: float = 0.20
    ):
        self.initial_elo = initial_elo
        self.initial_sigma = initial_sigma
        self.k_factor = k_factor
        self.ewma_alpha = ewma_alpha

    def calculate_features(
        self,
        historical_matches: List[Dict[str, Any]],
        active_matches: List[Dict[str, Any]],
        target_matchday: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        elo_dict: Dict[int, float] = {}
        matches_played: Dict[int, int] = {}
        
        home_att: Dict[int, float] = {}
        home_def: Dict[int, float] = {}
        away_att: Dict[int, float] = {}
        away_def: Dict[int, float] = {}

        # 1. Procesar histórico base (2025/2026)
        sorted_hist = sorted(historical_matches, key=lambda x: (x["season"], x["matchday"], x["date"]))
        hist_rows = self._process_matches(
            sorted_hist, elo_dict, matches_played, home_att, home_def, away_att, away_def
        )

        # Regresión a la media entre temporadas (15%)
        for t_id in elo_dict:
            elo_dict[t_id] = 0.85 * elo_dict[t_id] + 0.15 * self.initial_elo

        # 2. Procesar temporada activa hasta la jornada objetivo
        sorted_active = sorted(active_matches, key=lambda x: (x["matchday"], x["date"]))
        if target_matchday is not None:
            sorted_active = [m for m in sorted_active if m["matchday"] <= target_matchday]

        active_rows = self._process_matches(
            sorted_active, elo_dict, matches_played, home_att, home_def, away_att, away_def
        )

        return pd.DataFrame(hist_rows), pd.DataFrame(active_rows)

    def _process_matches(
        self,
        matches: List[Dict[str, Any]],
        elo_dict: Dict[int, float],
        matches_played: Dict[int, int],
        home_att: Dict[int, float],
        home_def: Dict[int, float],
        away_att: Dict[int, float],
        away_def: Dict[int, float]
    ) -> List[Dict[str, Any]]:
        rows = []

        for m in matches:
            h_id = m["home_team_id"]
            a_id = m["away_team_id"]

            for t_id in [h_id, a_id]:
                if t_id not in elo_dict:
                    elo_dict[t_id] = self.initial_elo
                    matches_played[t_id] = 0
                    home_att[t_id] = 1.45
                    home_def[t_id] = 1.10
                    away_att[t_id] = 1.05
                    away_def[t_id] = 1.40

            h_elo = elo_dict[h_id]
            a_elo = elo_dict[a_id]
            elo_diff = h_elo - a_elo

            home_factor = home_att[h_id] / max(0.5, away_att[h_id])

            h_sigma = max(40.0, self.initial_sigma / (1.0 + 0.08 * matches_played[h_id]))
            a_sigma = max(40.0, self.initial_sigma / (1.0 + 0.08 * matches_played[a_id]))

            row = {
                "match_id": m["id"],
                "season": m["season"],
                "matchday": m["matchday"],
                "date": m["date"],
                "home_team_id": h_id,
                "away_team_id": a_id,
                "home_elo": h_elo,
                "away_elo": a_elo,
                "elo_diff": elo_diff,
                "home_factor": home_factor,
                "home_sigma": h_sigma,
                "away_sigma": a_sigma,
                "home_avg_scored": home_att[h_id],
                "home_avg_conceded": home_def[h_id],
                "away_avg_scored": away_att[a_id],
                "away_avg_conceded": away_def[a_id],
                "target_home_score": m.get("home_score"),
                "target_away_score": m.get("away_score"),
                "status": m["status"]
            }
            rows.append(row)

            if m["status"] == "FINISHED" and m["home_score"] is not None and m["away_score"] is not None:
                h_score = float(m["home_score"])
                a_score = float(m["away_score"])
                alpha = self.ewma_alpha

                home_att[h_id] = (1 - alpha) * home_att[h_id] + alpha * h_score
                home_def[h_id] = (1 - alpha) * home_def[h_id] + alpha * a_score

                away_att[a_id] = (1 - alpha) * away_att[a_id] + alpha * a_score
                away_def[a_id] = (1 - alpha) * away_def[a_id] + alpha * h_score

                s_h = 1.0 if h_score > a_score else (0.5 if h_score == a_score else 0.0)
                e_h = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

                goal_diff = abs(h_score - a_score)
                margin_mult = math.log(goal_diff + 1) if goal_diff > 1 else 1.0

                elo_dict[h_id] = h_elo + (self.k_factor * margin_mult) * (s_h - e_h)
                elo_dict[a_id] = a_elo + (self.k_factor * margin_mult) * ((1.0 - s_h) - (1.0 - e_h))

                matches_played[h_id] += 1
                matches_played[a_id] += 1

        return rows