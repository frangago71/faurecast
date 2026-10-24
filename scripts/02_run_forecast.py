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
from src.services.predict_service import PredictService


def load_config():
    config_path = "config.yaml"
    if HAS_YAML and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "database": {"path": "data/faurecast.db"},
        "model": {"weights_path": "models/faurecast_nn.pt"}
    }


def main():
    parser = argparse.ArgumentParser(description="Genera pronósticos con Faurecast.")
    parser.add_argument("-j", "--matchday", type=int, default=None, help="Jornada a predecir.")
    args = parser.parse_args()

    print("==========================================================================================")
    print("  Faurecast - Generación de Pronósticos y Exportación a data/raw/")
    print("==========================================================================================")

    config = load_config()
    db_path = config["database"]["path"]
    model_path = config.get("model", {}).get("weights_path", "models/faurecast_nn.pt")

    engine = get_sqlite_engine(db_path)
    session = get_session_factory(engine)()

    try:
        service = PredictService(session, model_path=model_path)
        count, matchday, csv_path, rows = service.run_predictions(matchday=args.matchday)

        if rows:
            print(f"\n📌 PRONÓSTICOS JORNADA {matchday}:\n")
            header = f"{'FECHA':<17} | {'PARTIDO':<35} | {'xG (L-V)':<10} | {'EXACTO':<8} | {'1X2':<5} | {'CONF.':<7}"
            print(header)
            print("-" * len(header))

            for r in rows:
                match_str = f"{r['equipo_local']} vs {r['equipo_visitante']}"
                xg_str = f"{r['xg_local']:.2f} - {r['xg_visitante']:.2f}"
                print(f"{r['fecha']:<17} | {match_str:<35} | {xg_str:<10} | {r['marcador_exacto']:<8} | {r['pronostico_1x2']:<5} | {r['confianza']:<7}")

            print("\n==========================================================================================")
            print(f"✓ Registros procesados en BD: {len(rows)}")
            print(f"✓ Archivo CSV exportado en: {csv_path}")
            print("==========================================================================================")
        else:
            print(f"\n-> No se encontraron partidos para la jornada especificada.")

    except Exception as e:
        session.rollback()
        print(f"\n❌ ERROR al generar pronósticos: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()