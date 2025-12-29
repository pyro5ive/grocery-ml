import pandas as pd
class WeatherFeatures:  
    
    @staticmethod
    def BuildWeather(sourcePath):
        # --- WEATHER PREP ---
        weatherCols=["datetime", "temp", "humidity", "feelslike", "dew", "precip"]
        df_weather = pd.read_csv(sourcePath, usecols=weatherCols)
        df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
        df_weather = df_weather.set_index("datetime").sort_index()
        df_weather["temp_5day_avg_feat"] = df_weather["temp"].rolling(5, min_periods=1).mean()
        df_weather["feelsLike_5day_avg_feat"] = df_weather["feelslike"].rolling(5, min_periods=1).mean()
        df_weather["dew_5day_avg_feat"] = df_weather["dew"].rolling(5, min_periods=1).mean()
        df_weather["humidity_5day_avg_feat"] = df_weather["humidity"].rolling(5, min_periods=1).mean()
        df_weather["precip_5day_avg_feat"] = df_weather["precip"].rolling(5, min_periods=1).mean()
        df_weather = df_weather.drop(columns=["temp", "humidity", "feelslike", "dew", "precip"])
        # convert index to date for merging
        df_weather["date"] = df_weather.index.date
        df_weather["date"] = pd.to_datetime(df_weather["date"])
        df_weather = df_weather.set_index("date")
        return df_weather;
        #####################################################################################################