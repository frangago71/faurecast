import os
import sys

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Añadir el directorio raíz al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database.connection import get_sqlite_engine, get_session_factory, init_db
from src.database.models import Base, League, Team, Match
from src.data_loader.provider import LocalSimulatedMatchProvider


def load_config():
    config_path = "config.yaml"
    if HAS_YAML and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "database": {"path": "data/faurecast.db"},
        "competition": {
            "league_name": "LaLiga",
            "league_code": "ESP-LaLiga",
            "country": "Spain",
            "seasons": {"historical": "2025-2026", "active": "2026-2027"}
        }
    }


def main():
    print("==================================================")
    print("  Faurecast - Inicializacion de Base de Datos")
    print("==================================================")

    config = load_config()
    db_path = config["database"]["path"]
    print(f"-> Conectando a la base de datos SQLite: {db_path}")
    engine = get_sqlite_engine(db_path)

    print("-> Creando tablas...")
    init_db(engine)

    SessionFactory = get_session_factory(engine)
    session = SessionFactory()

    try:
        league_code = config["competition"]["league_code"]
        existing_league = session.query(League).filter_by(code=league_code).first()

        if existing_league:
            response = input(f"-> La liga '{league_code}' ya existe en la base de datos. ¿Deseas borrar y reescribir los datos? (s/N): ").strip().lower()
            if response in ["s", "si", "sí", "y", "yes"]:
                print("-> Eliminando datos existentes...")
                session.close()  # Cerramos la sesión activa antes de eliminar tablas
                Base.metadata.drop_all(engine)
                Base.metadata.create_all(engine)
                session = SessionFactory()  # Reabrimos la sesión limpia
                print("-> Tablas recreadas correctamente.")
            else:
                print("-> Operacion cancelada. Saltando sembrado.")
                return

        # 1. Registrar Liga
        league = League(
            name=config["competition"]["league_name"],
            code=league_code,
            country=config["competition"]["country"]
        )
        session.add(league)
        session.flush()
        print(f"OK: Liga registrada: {league.name} ({league.code})")

        # 2. Registrar Equipos
        provider = LocalSimulatedMatchProvider()
        teams_data = provider.get_teams()
        team_map = {}

        for t_data in teams_data:
            team = Team(
                league_id=league.id,
                name=t_data["name"],
                short_name=t_data["short_name"],
                code=t_data["code"]
            )
            session.add(team)
            session.flush()
            team_map[team.code] = team.id

        print(f"OK: {len(team_map)} equipos de LaLiga registrados.")

        # 3. Cargar e insertar datos historicos
        hist_season = config["competition"]["seasons"]["historical"]
        print(f"-> Cargando temporada historica {hist_season}...")
        hist_matches = provider.get_historical_matches(hist_season)

        for m_data in hist_matches:
            match = Match(
                league_id=league.id,
                season=m_data["season"],
                matchday=m_data["matchday"],
                date=m_data["date"],
                home_team_id=team_map[m_data["home_team_code"]],
                away_team_id=team_map[m_data["away_team_code"]],
                home_score=m_data["home_score"],
                away_score=m_data["away_score"],
                status=m_data["status"]
            )
            session.add(match)

        print(f"OK: {len(hist_matches)} partidos historicos ({hist_season}) insertados.")

        # 4. Cargar e insertar temporada activa
        act_season = config["competition"]["seasons"]["active"]
        print(f"-> Cargando temporada activa {act_season}...")
        active_matches = provider.get_active_matches(act_season)

        for m_data in active_matches:
            match = Match(
                league_id=league.id,
                season=m_data["season"],
                matchday=m_data["matchday"],
                date=m_data["date"],
                home_team_id=team_map[m_data["home_team_code"]],
                away_team_id=team_map[m_data["away_team_code"]],
                home_score=m_data["home_score"],
                away_score=m_data["away_score"],
                status=m_data["status"]
            )
            session.add(match)

        print(f"OK: {len(active_matches)} partidos de la temporada activa ({act_season}) insertados.")

        session.commit()
        print("==================================================")
        print("  OK: Base de datos inicializada correctamente.")
        print("==================================================")

    except Exception as e:
        session.rollback()
        print(f"ERROR al inicializar la base de datos: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()