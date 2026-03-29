from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Any


class WeatherServiceBase(ABC):
    """
    Base interface for weather data providers.

    Implementations are responsible for retrieving current conditions
    and historical or forecast weather data for a given latitude,
    longitude, and date.
    """

    @abstractmethod
    def get_current_conditions(
        self,
        latitude: float,
        longitude: float
    ) -> dict[str, Any]:
        """
        Retrieve current weather conditions.

        :param latitude: Geographic latitude.
        :param longitude: Geographic longitude.
        :returns: Dictionary containing current weather metrics.
        """
        raise NotImplementedError
    #--------------------------#

    @abstractmethod
    def get_forecast_by_date(
        self,
        latitude: float,
        longitude: float,
        target_date: date | datetime
    ) -> list[dict[str, Any]]:
        """
        Retrieve forecast weather data for a specific date.

        :param latitude: Geographic latitude.
        :param longitude: Geographic longitude.
        :param target_date: Target date for the forecast.
        :returns: List of forecast period dictionaries.
        """
        raise NotImplementedError
    #--------------------------#

    @abstractmethod
    def get_weather_for_date(
        self,
        latitude: float,
        longitude: float,
        date_value: date | datetime | str
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Retrieve weather data for a given date, automatically selecting
        between current conditions and forecast data.

        :param latitude: Geographic latitude.
        :param longitude: Geographic longitude.
        :param date_value: Date value (date, datetime, or ISO string).
        :returns: Weather data for the requested date.
        """
        raise NotImplementedError
    #--------------------------#
