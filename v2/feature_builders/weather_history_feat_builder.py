import logging
import pandas as pd

class WeatherHistory_FeatureBuilder:

    sourcePath = "../data/weather/VisualCrossing-70062 2000-01-01 to 2026-23-1.csv";

    requiredFeatures = [];
    producedFeatures = ["feelsLike_feat","humidity_feat","precip_feat"];

    def __init__(this, dateCol: str = "date"):
        this.logger = logging.getLogger(this.__class__.__name__);
        this.dateCol = dateCol;
    #-----------------------------------------------------------------#
    def build_feature(this, df):
        this._check_reqs(df);
        this.weatherDf = this._load_weather_df();
        outDf = df.copy();
        outDf[this.dateCol] = pd.to_datetime(outDf[this.dateCol]).dt.normalize();
        return outDf.merge(this.weatherDf, left_on=this.dateCol, right_index=True, how="left");
    #-----------------------------------------------------------------#
    
    def _check_reqs(this, df):
        if this.dateCol not in df.columns:
            raise RuntimeError(f"missing required date column {this.dateCol}");
    #-----------------------------------------------------------------#
    def _load_weather_df(this):
        df = pd.read_csv(
            WeatherHistory_FeatureBuilder.sourcePath,
            usecols=[
                "datetime",
                # "temp",
                "feelslike",
                "humidity",
                "precip"
                # "windspeed",
                # "sealevelpressure"
            ]
         );
        df["datetime"] = pd.to_datetime(df["datetime"]);
        df["date"] = df["datetime"].dt.normalize();
        df = df.rename(columns={
            # "temp":"temp_feat",
            "feelslike":"feelsLike_feat",
            "humidity":"humidity_feat",
            "precip":"precip_feat",
            # "windspeed":"windspeed_feat",
            # "sealevelpressure":"sealevelpressure_feat"
        });
        df = df.drop(columns=["datetime"]).groupby("date", as_index=True).mean();
        return df;
    #-----------------------------------------------------------------#
  
