"""
API helper functions for the MCP server.

This module contains utilities for processing API responses.
"""

from typing import Dict, Any

def process_weather_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process raw weather API data into a more structured format.
    
    Parameters:
    -----------
    data : Dict[str, Any]
        Raw weather data from the API
        
    Returns:
    --------
    Dict[str, Any]
        Processed weather data
    """
    # Process the data to provide a better structure
    if 'current_weather' in data:
        current = data['current_weather']
        return {
            "temperature": current.get('temperature'),
            "wind_speed": current.get('windspeed'),
            "wind_direction": current.get('winddirection'),
            "weather_code": current.get('weathercode'),
            "is_day": current.get('is_day'),
            "timestamp": current.get('time')
        }
    return data


def process_fixture_data(data: Dict[str, Any]) -> Dict[str, Any]:

    from utils.models import FixturesAPIResponse

    fixtures = FixturesAPIResponse.model_validate(data)

    result = ""

    # Access the upcoming matches
    for fixture in fixtures.response:
        match_date = fixture.fixture.date
        home_team = fixture.teams.home.name
        away_team = fixture.teams.away.name
        venue = fixture.fixture.venue.name
        
        result += f"{match_date.strftime('%d %b %Y, %H:%M')}: {home_team} vs {away_team} at {venue}\n"
        # print(f"{match_date.strftime('%d %b %Y, %H:%M')}: {home_team} vs {away_team} at {venue}")
    
    return result