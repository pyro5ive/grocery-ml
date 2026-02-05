import pandas as pd
import logging

logger = logging.getLogger(__name__)
class WeatherFeatures:  

    sourcePath = "data\weather\VisualCrossing-70062 2000-01-01 to 2026-23-1.csv"
    
    @staticmethod
    def create_weather_df():
        # --- WEATHER PREP ---
        weatherCols=["datetime", "temp", "humidity", "feelslike", "precip"]
        df_weather = pd.read_csv(sourcePath, usecols=weatherCols)
        df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
        df_weather = df_weather.set_index("datetime").sort_index()
        df_weather["temp_5day_avg_feat"] = df_weather["temp"].rolling(5, min_periods=1).mean()
        df_weather["feelsLike_5day_avg_feat"] = df_weather["feelslike"].rolling(5, min_periods=1).mean()
        df_weather["humidity_5day_avg_feat"] = df_weather["humidity"].rolling(5, min_periods=1).mean()
        df_weather["precip_5day_avg_feat"] = df_weather["precip"].rolling(5, min_periods=1).mean()
        df_weather = df_weather.drop(columns=["temp", "humidity", "feelslike", "dew", "precip"])
        # convert index to date for merging
        df_weather["date"] = df_weather.index.date
        df_weather["date"] = pd.to_datetime(df_weather["date"])
        df_weather = df_weather.set_index("date")
        return df_weather;
    #####################################################################################################
    @staticmethod
    def create_weather_df_daily():
        df_weather = pd.read_csv(
            WeatherFeatures.sourcePath,
            usecols=["datetime", "temp", "feelslike", "humidity", 
                     "precip", "windspeed","sealevelpressure",
                     # "sunrise",# "sunset"
                    ]
        )
    
        df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
        # df_weather["sunrise_datetime_raw"] = pd.to_datetime(df_weather["sunrise"])
        # df_weather["sunset_datetime_raw"] = pd.to_datetime(df_weather["sunset"])
        # df_weather = df_weather.drop(columns=["sunrise", "sunset"])
    
        df_weather = df_weather.rename(columns={
            "temp": "temp_feat",
            "feelslike": "feelsLike_feat",
            "humidity": "humidity_feat",
            "precip": "precip_feat",
            "windspeed": "windspeed_feat",
            "sealevelpressure": "sealevelpressure_feat"
        })
    
        return df_weather
    ######################################################################################################
    @staticmethod
    def merge_weather_features(weatherDf, targetDf):
        weatherDf = weatherDf.copy()
        targetDf = targetDf.copy()
    
        weatherDf["date"] = weatherDf["datetime"].dt.normalize()
        targetDf["date"] = targetDf["date"].dt.normalize()
    
        weatherMinDate = weatherDf["datetime"].min()
        weatherMaxDate = weatherDf["datetime"].max()
    
        targetMinDate = targetDf["date"].min()
        targetMaxDate = targetDf["date"].max()
    
        if targetMinDate < weatherMinDate or targetMaxDate > weatherMaxDate:
            logger.warn("dates are missing in history")
            
            # raise ValueError(
            #     f"Target date range [{targetMinDate}, {targetMaxDate}] "
            #     f"exceeds weather date range [{weatherMinDate}, {weatherMaxDate}]"
            # )
    
        mergedDf = targetDf.merge(
            weatherDf.drop(columns=["datetime"]),
            on="date",
            how="left"
        )
    
        # mergedDf = mergedDf.drop(columns=["date"])
    
        return mergedDf
    ######################################################################################################

