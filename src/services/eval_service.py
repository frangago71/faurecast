import math
import numpy as np
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from src.database.models import Match, Prediction


class EvalService:
    def __init__(self, session: Session):
        self.session = session

    def evaluate_predictions(self, matchday: Optional[int] = None) -> Dict[str, Any]:
        query = (
            self.session.query(Match, Prediction)
            .join(Prediction, Match.id == Prediction.match_id)
            .filter(Match.status == "FINISHED")
        )

        if matchday is not None:
            query = query.filter(Match.matchday == matchday)

        results = query.all()

        if not results:
            return {"total": 0, "matchday": matchday, "rows": []}

        rows = []
        hits_1x2 = 0
        exact_score_hits = 0
        home_score_errors = []
        away_score_errors = []

        for match, pred in results:
            # Signo real
            if match.home_score > match.away_score:
                actual_1x2 = "1"
            elif match.home_score < match.away_score:
                actual_1x2 = "2"
            else:
                actual_1x2 = "X"

            is_hit_1x2 = (pred.predicted_1x2 == actual_1x2)
            if is_hit_1x2:
                hits_1x2 += 1

            # Reconstruccion condicional del marcador exacto previsto
            l_h = pred.predicted_home_score
            l_a = pred.predicted_away_score
            max_g = 6

            p_matrix = np.zeros((max_g + 1, max_g + 1))
            for i in range(max_g + 1):
                p_i = (l_h ** i) * math.exp(-l_h) / math.factorial(i)
                for j in range(max_g + 1):
                    p_j = (l_a ** j) * math.exp(-l_a) / math.factorial(j)
                    p_matrix[i, j] = p_i * p_j

            mask = np.zeros_like(p_matrix, dtype=bool)
            if pred.predicted_1x2 == "1":
                mask[np.tril_indices(max_g + 1, -1)] = True
            elif pred.predicted_1x2 == "2":
                mask[np.triu_indices(max_g + 1, 1)] = True
            else:
                np.fill_diagonal(mask, True)

            masked_p_matrix = np.where(mask, p_matrix, -1.0)
            exact_pred_h, exact_pred_a = np.unravel_index(np.argmax(masked_p_matrix), p_matrix.shape)

            is_exact_hit = (exact_pred_h == match.home_score) and (exact_pred_a == match.away_score)
            if is_exact_hit:
                exact_score_hits += 1

            # Simbolos limpios: ✓ = Pleno, ~ = Acierto 1X2, X = Fallo
            if is_hit_1x2 and is_exact_hit:
                status_symbol = "✓"
            elif is_hit_1x2:
                status_symbol = "~"
            else:
                status_symbol = "X"

            err_h = abs(pred.predicted_home_score - match.home_score)
            err_a = abs(pred.predicted_away_score - match.away_score)
            home_score_errors.append(err_h)
            away_score_errors.append(err_a)

            rows.append({
                "jornada": match.matchday,
                "partido": f"{match.home_team.name} vs {match.away_team.name}",
                "real": f"{match.home_score}-{match.away_score} ({actual_1x2})",
                "previsto": f"{exact_pred_h}-{exact_pred_a} ({pred.predicted_1x2})",
                "pred_xg": f"{pred.predicted_home_score:.2f}-{pred.predicted_away_score:.2f}",
                "confianza": f"{pred.confidence * 100:.1f}%",
                "acierto": status_symbol
            })

        total = len(results)
        return {
            "total": total,
            "matchday": matchday,
            "acc_1x2": (hits_1x2 / total) * 100,
            "acc_exact": (exact_score_hits / total) * 100,
            "mae_home": float(np.mean(home_score_errors)),
            "mae_away": float(np.mean(away_score_errors)),
            "mae_global": float(np.mean(home_score_errors + away_score_errors)),
            "rows": rows
        }