from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, Integer, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Clase base declarativa de SQLAlchemy para Faurecast."""
    pass


class League(Base):
    __tablename__ = "leagues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    code: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    country: Mapped[str] = mapped_column(String(50), nullable=False)

    teams: Mapped[List["Team"]] = relationship("Team", back_populates="league", cascade="all, delete-orphan")
    matches: Mapped[List["Match"]] = relationship("Match", back_populates="league", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<League(code='{self.code}', name='{self.name}')>"


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    short_name: Mapped[str] = mapped_column(String(50), nullable=False)
    code: Mapped[str] = mapped_column(String(10), unique=True, nullable=False)

    league: Mapped["League"] = relationship("League", back_populates="teams")
    home_matches: Mapped[List["Match"]] = relationship(
        "Match", foreign_keys="Match.home_team_id", back_populates="home_team"
    )
    away_matches: Mapped[List["Match"]] = relationship(
        "Match", foreign_keys="Match.away_team_id", back_populates="away_team"
    )

    def __repr__(self) -> str:
        return f"<Team(code='{self.code}', name='{self.name}')>"


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False)
    season: Mapped[str] = mapped_column(String(20), nullable=False)
    matchday: Mapped[int] = mapped_column(Integer, nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)

    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="SCHEDULED", nullable=False)  # SCHEDULED, FINISHED

    league: Mapped["League"] = relationship("League", back_populates="matches")
    home_team: Mapped["Team"] = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team: Mapped["Team"] = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")

    predictions: Mapped[List["Prediction"]] = relationship(
        "Prediction", back_populates="match", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Match(id={self.id}, season='{self.season}', J{self.matchday}: "
            f"{self.home_team_id} vs {self.away_team_id}, status='{self.status}')>"
        )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"), nullable=False)

    predicted_home_score: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_away_score: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_1x2: Mapped[str] = mapped_column(String(1), nullable=False)  # '1', 'X', '2'
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, default="v1.0")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    match: Mapped["Match"] = relationship("Match", back_populates="predictions")
    evaluation: Mapped[Optional["Evaluation"]] = relationship(
        "Evaluation", back_populates="prediction", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction(match_id={self.match_id}, "
            f"pred={self.predicted_home_score:.1f}-{self.predicted_away_score:.1f} [{self.predicted_1x2}])>"
        )


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False)

    actual_1x2: Mapped[str] = mapped_column(String(1), nullable=False)  # '1', 'X', '2'
    is_1x2_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_exact_score: Mapped[bool] = mapped_column(Boolean, nullable=False)

    home_score_error: Mapped[float] = mapped_column(Float, nullable=False)
    away_score_error: Mapped[float] = mapped_column(Float, nullable=False)

    points_earned: Mapped[int] = mapped_column(Integer, nullable=False)  # e.g., +3 por exacto, +1 por 1X2
    evaluated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="evaluation")

    def __repr__(self) -> str:
        return f"<Evaluation(pred_id={self.prediction_id}, points={self.points_earned}, exact={self.is_exact_score})>"
