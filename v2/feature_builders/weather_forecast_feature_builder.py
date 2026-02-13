import logging
import pandas as pd
from datetime import datetime
from services.weather.weather_service import NwsWeatherService


class WeatherForecastFeatureBuilder:

    def __init__(self, weather_service: NwsWeatherService, latitude: float, longitude: float):
        self.logger = logging.getLogger(__name__)
        self.weather_service = weather_service
        self.latitude = latitude
        self.longitude = longitude

    # ============================================================
    def build_df(self, df: pd.DataFrame, prediction_date: datetime) -> pd.DataFrame:

        self.logger.info(
            "WeatherForecastFeatureBuilder start rows=%s date=%s",
            len(df),
            prediction_date
        )

        df = df.copy()
        dateString = str(prediction_date)
        weather = self.weather_service.get_weather_for_date(self.latitude, self.longitude, dateString)

        precip = 0.0
        humidity = None
        temp_c = None

        if isinstance(weather, dict):
            temp_c = weather.get("temp_c")
            humidity = weather.get("humidity_pct")
            precip = weather.get("rain_mm_last_hour", 0.0)

        elif isinstance(weather, list) and len(weather) > 0:
            forecast = weather[0]
            temp_f = forecast.get("temp_f")
            humidity = forecast.get("humidity_pct")
            temp_c = (temp_f - 32) * 5.0 / 9.0 if temp_f is not None else None

        feels_like = self._compute_feels_like_c(temp_c, humidity)

        df["precip_cont"] = precip
        df["humidity_cont"] = humidity
        df["feelsLike_cont"] = feels_like

        self.logger.info("Weather features applied precip=%s humidity=%s feels_like=%s", precip, humidity, feels_like )

        return df

    # ============================================================
    def _compute_feels_like_c(self, temp_c, humidity_pct):

        if temp_c is None or humidity_pct is None:
            return None

        if temp_c >= 27:
            t = temp_c
            rh = humidity_pct
            hi = (
                -8.784695 +
                1.61139411 * t +
                2.338549 * rh +
                -0.14611605 * t * rh +
                -0.012308094 * t * t +
                -0.016424828 * rh * rh +
                0.002211732 * t * t * rh +
                0.00072546 * t * rh * rh +
                -0.000003582 * t * t * rh * rh
            )
            return hi

        return temp_c
