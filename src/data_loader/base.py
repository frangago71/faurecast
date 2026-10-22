from abc import ABC, abstractmethod
from typing import List, Dict, Any


class MatchDataProvider(ABC):
    """Clase base abstracta para el proveedor de datos de partidos."""

    @abstractmethod
    def get_teams(self) -> List[Dict[str, Any]]:
        """Retorna la lista de equipos de la competición con metadatos."""
        pass

    @abstractmethod
    def get_historical_matches(self, season: str) -> List[Dict[str, Any]]:
        """Retorna los partidos históricos finalizados para la temporada de calibración."""
        pass

    @abstractmethod
    def get_active_matches(self, season: str) -> List[Dict[str, Any]]:
        """Retorna los partidos (finalizados y programados) para la temporada activa."""
        pass