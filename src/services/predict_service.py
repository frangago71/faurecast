import os
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from sqlalchemy.orm import Session

from src.database.models import Match, Prediction
from src.features.feature_builder import FeatureBuilder
from src.models.neural_network import NeuralNetworkModel


class PredictService:
    def __init__(self, session: Session, model_path: str = "models/faurecast_nn.pt"):
        self.session = session
        self.feature_builder = FeatureBuilder()
        self.nn_model = NeuralNetworkModel(weights_path=model_path)

    def run_predictions(self, matchday: Optional[int] = None) -> Tuple[int, int, str, List[Dict[str, Any]]]:
        all_matches = self.session.query(Match).order_by(Match.season, Match.matchday, Match.date).all()
        
        hist_matches = [self._match_to_dict(m) for m in all_matches if m.season == "2025-2026"]
        act_matches = [self._match_to_dict(m) for m in all_matches if m.season == "2026-2027"]

        # 1. Determinar jornada a predecir
        if matchday is None:
            pending = [m for m in act_matches if m["status"] == "SCHEDULED"]
            target_matchday = min([m["matchday"] for m in pending]) if pending else max([m["matchday"] for m in act_matches])
        else:
            target_matchday = matchday

        # 2. Calcular características de forma incremental hasta la jornada solicitada
        df_hist, df_act = self.feature_builder.calculate_features(
            hist_matches, act_matches, target_matchday=target_matchday
        )

        # 3. Entrenar el modelo con partidos ya disputados
        df_train = pd.concat([df_hist, df_act[df_act["status"] == "FINISHED"]]).dropna(
            subset=["target_home_score", "target_away_score"]
        )

        if not df_train.empty:
            X_train = df_train[NeuralNetworkModel.FEATURE_COLS].values
            y_train = df_train[["target_home_score", "target_away_score"]].values
            self.nn_model.fit(X_train, y_train, epochs=100)

        # 4. Predecir partidos de la jornada objetivo
        target_matches_df = df_act[df_act["matchday"] == target_matchday]
        if target_matches_df.empty:
            return 0, target_matchday, "", []

        X_target = target_matches_df[NeuralNetworkModel.FEATURE_COLS].values
        details = self.nn_model.predict_match_details(X_target)

        created_count = 0
        csv_rows = []

        for idx, (_, row) in enumerate(target_matches_df.iterrows()):
            match_id = int(row["match_id"])
            det = details[idx]

            existing = self.session.query(Prediction).filter_by(match_id=match_id).first()
            if not existing:
                prediction = Prediction(
                    match_id=match_id,
                    predicted_home_score=det["xg_home"],
                    predicted_away_score=det["xg_away"],
                    predicted_1x2=det["predicted_1x2"],
                    confidence=det["confidence"],
                    model_version="poisson_nn_v1.0"
                )
                self.session.add(prediction)
                created_count += 1
            else:
                existing.predicted_home_score = det["xg_home"]
                existing.predicted_away_score = det["xg_away"]
                existing.predicted_1x2 = det["predicted_1x2"]
                existing.confidence = det["confidence"]

            m_obj = next(m for m in act_matches if m["id"] == match_id)
            csv_rows.append({
                "jornada": target_matchday,
                "fecha": m_obj["date"].strftime("%Y-%m-%d %H:%M"),
                "equipo_local": m_obj["home_team_name"],
                "equipo_visitante": m_obj["away_team_name"],
                "xg_local": round(det["xg_home"], 2),
                "xg_visitante": round(det["xg_away"], 2),
                "marcador_exacto": det["exact_score"],
                "pronostico_1x2": det["predicted_1x2"],
                "confianza": f"{det['confidence'] * 100:.1f}%"
            })

        self.session.commit()

        # 5. Exportar CSV a data/raw/
        raw_dir = "data/raw"
        os.makedirs(raw_dir, exist_ok=True)
        csv_path = os.path.join(raw_dir, f"jornada_{target_matchday}.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

        return created_count, target_matchday, csv_path, csv_rows

    def _match_to_dict(self, m: Match) -> Dict[str, Any]:
        return {
            "id": m.id,
            "season": m.season,
            "matchday": m.matchday,
            "date": m.date,
            "home_team_id": m.home_team_id,
            "away_team_id": m.away_team_id,
            "home_score": m.home_score,
            "away_score": m.away_score,
            "status": m.status,
            "home_team_name": m.home_team.name,
            "away_team_name": m.away_team.name
        }