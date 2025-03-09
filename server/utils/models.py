from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class FixtureStatus(BaseModel):
    long: str
    short: str
    elapsed: Optional[int] = None
    extra: Optional[Any] = None


class FixturePeriods(BaseModel):
    first: Optional[int] = None
    second: Optional[int] = None


class FixtureVenue(BaseModel):
    id: int
    name: str
    city: str


class Fixture(BaseModel):
    id: int
    referee: Optional[str] = None
    timezone: str
    date: datetime
    timestamp: int
    periods: FixturePeriods
    venue: FixtureVenue
    status: FixtureStatus


class Team(BaseModel):
    id: int
    name: str
    logo: str
    winner: Optional[bool] = None


class Teams(BaseModel):
    home: Team
    away: Team


class Goals(BaseModel):
    home: Optional[int] = None
    away: Optional[int] = None


class ScoreDetail(BaseModel):
    home: Optional[int] = None
    away: Optional[int] = None


class Score(BaseModel):
    halftime: ScoreDetail
    fulltime: ScoreDetail
    extratime: ScoreDetail
    penalty: ScoreDetail


class League(BaseModel):
    id: int
    name: str
    country: str
    logo: str
    flag: str
    season: int
    round: str
    standings: bool


class FixtureResponse(BaseModel):
    fixture: Fixture
    league: League
    teams: Teams
    goals: Goals
    score: Score


class FixturesAPIResponse(BaseModel):
    get: str
    parameters: Dict[str, Any]
    errors: List[Any]
    results: int
    paging: Dict[str, int]
    response: List[FixtureResponse]