import requests
import logging
from datetime import datetime, date, timedelta

logger = logging.getLogger(__name__)

class NwsWeatherService:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "(grocery-ml, nolabizit@gmail.com)"})
        self.points_url = "https://api.weather.gov/points"

        self.cached_periods = None
        self.cached_latest = None

        self.cached_current = None
        self.cached_current_time = None
        self.current_ttl = timedelta(minutes=10)

        logger.info("NwsWeatherService initialized")
#############################################
    def _load_forecast_cache(self, latitude, longitude):
        logger.info("Loading forecast cache lat=%s lon=%s", latitude, longitude)

        url = f"{self.points_url}/{latitude},{longitude}"
        r = self.session.get(url, timeout=10)
        forecast_url = r.json()["properties"]["forecast"]

        r2 = self.session.get(forecast_url, timeout=10)
        periods = r2.json()["properties"]["periods"]

        self.cached_periods = periods
        self.cached_latest = max(
            datetime.fromisoformat(p["startTime"]).date()
            for p in periods
        )

        logger.info(
            "Forecast cache loaded periods=%s latest_date=%s",
            len(periods),
            self.cached_latest
        )
#############################################
    def _get_forecast_periods(self, latitude, longitude):
        if self.cached_periods is None:
            logger.info("Forecast cache empty, loading")
            self._load_forecast_cache(latitude, longitude)
        return self.cached_periods
#############################################
    def get_current_conditions(self, latitude, longitude):
        if (
            self.cached_current is not None
            and datetime.now() - self.cached_current_time < self.current_ttl
        ):
            logger.info("Returning cached current conditions")
            return self.cached_current

        logger.info("Fetching current conditions lat=%s lon=%s", latitude, longitude)

        url = f"{self.points_url}/{latitude},{longitude}"
        r = self.session.get(url, timeout=10)
        obs_url = r.json()["properties"]["observationStations"]

        r2 = self.session.get(obs_url, timeout=10)
        station_id = r2.json()["features"][0]["properties"]["stationIdentifier"]

        latest_url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
        r3 = self.session.get(latest_url, timeout=10)
        latest = r3.json()["properties"]

        precip = latest.get("precipitationLastHour")
        rain_mm_last_hour = (
            precip.get("value") if precip and precip.get("value") is not None else 0.0
        )

        self.cached_current = {
            "temp_c": latest["temperature"]["value"],
            "humidity_pct": latest["relativeHumidity"]["value"],
            "rain_mm_last_hour": rain_mm_last_hour
        }
        self.cached_current_time = datetime.now()

        logger.info(
            "Current conditions temp_c=%s humidity_pct=%s precip_mm=%s",
            self.cached_current["temp_c"],
            self.cached_current["humidity_pct"],
            rain_mm_last_hour
        )

        return self.cached_current
#############################################
    def get_forecast_by_date(self, latitude, longitude, target_date):
        logger.info("Fetching forecast for date=%s", target_date)

        periods = self._get_forecast_periods(latitude, longitude)

        if self.cached_latest is None or target_date > self.cached_latest:
            logger.info("Target date beyond cache, reloading forecast cache")
            self._load_forecast_cache(latitude, longitude)
            periods = self.cached_periods

        matched = []
        for p in periods:
            pd = datetime.fromisoformat(p["startTime"]).date()
            if pd == target_date:
                rh = p.get("relativeHumidity")
                humidity_pct = rh.get("value") if rh and rh.get("value") is not None else None

                matched.append({
                    "name": p["name"],
                    "temp_f": p["temperature"],
                    "humidity_pct": humidity_pct,
                    "short_forecast": p["shortForecast"]
                })

        logger.info("Forecast matches found=%s", len(matched))
        return matched
#############################################
    def get_weather_for_date(self, latitude, longitude, date_str):
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        logger.info("Weather request for date=%s", target_date)

        if target_date == date.today():
            logger.info("Date is today, using current conditions")
            return self.get_current_conditions(latitude, longitude)

        return self.get_forecast_by_date(latitude, longitude, target_date)
#############################################


class WeatherConditions:
    def __init__(self, date_value, temp_c, humidity_pct, precip_mm, short_forecast):
        self.date_value = date_value
        self.temp_c = temp_c
        self.humidity_pct = humidity_pct
        self.precip_mm = precip_mm
        self.short_forecast = short_forecast
#############################################
