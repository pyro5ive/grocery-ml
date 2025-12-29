import requests
from datetime import datetime, date, timedelta

class NwsWeatherService:
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "(myapp, contact@example.com)"})
        self.points_url = "https://api.weather.gov/points"

        self.cached_periods = None
        self.cached_latest = None

        self.cached_current = None
        self.cached_current_time = None
        self.current_ttl = timedelta(minutes=10)
    ###########################################################################################
    def _load_forecast_cache(self, latitude, longitude):
        url = f"{self.points_url}/{latitude},{longitude}"
        r = self.session.get(url, timeout=10)
        forecast_url = r.json()["properties"]["forecast"]

        r2 = self.session.get(forecast_url, timeout=10)
        periods = r2.json()["properties"]["periods"]

        self.cached_periods = periods
        self.cached_latest = max(datetime.fromisoformat(p["startTime"]).date() for p in periods)
    ##########################################################################################
    def _get_forecast_periods(self, latitude, longitude):
        if self.cached_periods is None:
            self._load_forecast_cache(latitude, longitude)
        return self.cached_periods
    ##########################################################################################
    def get_current_conditions(self, latitude, longitude):
        if self.cached_current is not None and datetime.now() - self.cached_current_time < self.current_ttl:
            return self.cached_current

        url = f"{self.points_url}/{latitude},{longitude}"
        r = self.session.get(url, timeout=10)
        obs_url = r.json()["properties"]["observationStations"]

        r2 = self.session.get(obs_url, timeout=10)
        station_id = r2.json()["features"][0]["properties"]["stationIdentifier"]

        latest_url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
        r3 = self.session.get(latest_url, timeout=10)
        latest = r3.json()["properties"]

        self.cached_current = {
            "temp_c": latest["temperature"]["value"],
            "humidity_pct": latest["relativeHumidity"]["value"],
            "rain_mm_last_hour": latest["precipitationLastHour"]["value"]
        }
        self.cached_current_time = datetime.now()
        return self.cached_current
    ###########################################################################################
    def get_forecast_by_date(self, latitude, longitude, target_date):
        periods = self._get_forecast_periods(latitude, longitude)

        if target_date > self.cached_latest:
            self._load_forecast_cache(latitude, longitude)
            periods = self.cached_periods

        matched = []
        for p in periods:
            pd = datetime.fromisoformat(p["startTime"]).date()
            if pd == target_date:
                matched.append({
                    "name": p["name"],
                    "temp_f": p["temperature"],
                    "humidity_pct": p.get("relativeHumidity"),
                    "short_forecast": p["shortForecast"]
                })
        return matched
    ##########################################################################################
    def get_weather_for_date(self, latitude, longitude, date_str):
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if target_date == date.today():
            return self.get_current_conditions(latitude, longitude)
        return self.get_forecast_by_date(latitude, longitude, target_date)
