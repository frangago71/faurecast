import os
import sys
import argparse

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database.connection import get_sqlite_engine, get_session_factory
from src.services.eval_service import EvalService


def load_config():
    config_path = "config.yaml"
    if HAS_YAML and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {"database": {"path": "data/faurecast.db"}}


def main():
    parser = argparse.ArgumentParser(description="Evalua los pronosticos de Faurecast.")
    parser.add_argument("-j", "--matchday", type=int, default=None, help="Jornada especifica a evaluar.")
    args = parser.parse_args()

    print("====================================================================================================")
    print("  Faurecast - Evaluacion de Rendimiento del Modelo")
    print("====================================================================================================")

    config = load_config()
    db_path = config["database"]["path"]

    engine = get_sqlite_engine(db_path)
    session = get_session_factory(engine)()

    try:
        service = EvalService(session)
        metrics = service.evaluate_predictions(matchday=args.matchday)

        if metrics["total"] == 0:
            print("\n-> No hay predicciones inmovilizadas para partidos finalizados en el filtro seleccionado.")
            return

        j_str = f"JORNADA {metrics['matchday']}" if metrics['matchday'] else "TODAS LAS JORNADAS DISPUTADAS"
        print(f"\nRESULTADOS DE EVALUACION ({j_str}):\n")

        header = f"{'PARTIDO':<38} | {'REAL':<10} | {'PREVISTO':<10} | {'xG PRED':<11} | {'CONF.':<7} | {'ESTADO':<6}"
        print(header)
        print("-" * len(header))

        for r in metrics["rows"]:
            print(f"{r['partido']:<38} | {r['real']:<10} | {r['previsto']:<10} | {r['pred_xg']:<11} | {r['confianza']:<7} | {r['acierto']:^6}")

        print("====================================================================================================")
        print(f"METRICAS GENERALES ({metrics['total']} partidos evaluados):")
        print(f"  - Precision Signo 1X2:       {metrics['acc_1x2']:.2f}%")
        print(f"  - Marcador Exacto Integro:  {metrics['acc_exact']:.2f}%")
        print(f"  - MAE Goles Local:           {metrics['mae_home']:.3f} goles")
        print(f"  - MAE Goles Visitante:       {metrics['mae_away']:.3f} goles")
        print(f"  - Error Absoluto Medio:      {metrics['mae_global']:.3f} goles")
        print("====================================================================================================")

    except Exception as e:
        print(f"\nERROR durante la evaluacion: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()