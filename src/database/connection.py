import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from src.database.models import Base


def get_sqlite_engine(db_path: str = "data/faurecast.db", echo: bool = False):
    """Crea y retorna un Engine de SQLAlchemy para SQLite con soporte FK habilitado."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db_url = f"sqlite:///{db_path}"
    
    engine = create_engine(db_url, echo=echo, connect_args={"check_same_thread": False})

    # Habilitar claves foráneas en SQLite (por defecto desactivadas)
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def get_session_factory(engine) -> sessionmaker[Session]:
    """Retorna una fábrica de sesiones SQLAlchemy."""
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db(engine) -> None:
    """Crea todas las tablas definidas en los modelos ORM."""
    Base.metadata.create_all(engine)