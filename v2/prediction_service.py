import logging
import pandas as pd
import os
import numpy as np

from model_repo.model_bundle import ModelBundle
from prediction_input_df_builder import PredictionInputDfBuilder
from feature_normalizer.continous_feature_normalizer import ContinousFeatureNormalizer
from feature_builders.weather_forecast_feature_builder import  WeatherForecastFeatureBuilder
from services.weather.weather_service import NwsWeatherService
from feature_schema import FeatureSchema
from datetime import datetime

class PredictionService:

    trainingSources = {
        "walmart": r"..\data\training\walmart",
        "winndixie": r"..\data\training\winndixie\txt",
        "winndixieAdditional": r"..\data\training\winndixie\additionalTxtRcpts",
        "weather": r"..\data\weather\VisualCrossing-70062 2000-01-01 to 2026-23-1.csv"
    }

    liveSources = {
        "walmart": r"..\data\live\walmart",
        "winndixie": r"..\data\live\winndixie\txt",
        "winndixieAdditional": r"..\data\live\winndixie\additionalTxtRcpts",
        "weather": r"..\date\weather\VisualCrossing-70062 2000-01-01 to 2026-23-1.csv"
    }

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__);
        self.predictionInputEventsDfBuilder = PredictionInputDfBuilder(self.liveSources, self.trainingSources);
        self.continuousFeatureNormalizer = ContinousFeatureNormalizer();
        self.featureSchema = FeatureSchema();

        self.weatherService = NwsWeatherService();
        self.weatherForcastFeatureBuilder = WeatherForecastFeatureBuilder(self.weatherService, 29.9934, -90.2580);
    #=================================================================================#

    def run_prediction(self, kerasModelBundle: ModelBundle, predictionDate: datetime):

        predInputNormDf = self._build_predict_input_df(kerasModelBundle, predictionDate);
        predictionsDf = self._predict(predInputNormDf, kerasModelBundle );
        return predictionsDf;
    # =================================================================================#
    def _predict(self, inputDfNorm: pd.DataFrame, modelBundle: ModelBundle) -> pd.DataFrame:

        featCols = self.featureSchema.get_feature_cols(inputDfNorm);
        x_features = inputDfNorm[featCols].to_numpy(np.float32)
        x_item_idx = inputDfNorm["itemId"].to_numpy(np.int32)

        # inputDfNorm = self.itemIdMapper.map_item_ids_to_names(inputDfNorm)
        prediction_values_col = modelBundle.model.predict(x_features, x_item_idx);
        inputDfNorm.insert(3, "readyToBuy_proabability", prediction_values_col)
        predDf = inputDfNorm.sort_values("readyToBuy_proabability", ascending=False).reset_index(drop=True)
        return predDf;
    # =================================================================================#
    def _build_predict_input_df(self, modelBundle: ModelBundle, predDate: datetime) -> pd.DataFrame:

        predInputDf = self.predictionInputEventsDfBuilder.build_df(predDate);
        contCols = self.featureSchema.get_continuous_cols(predInputDf);
        predInputNormDf = self.continuousFeatureNormalizer.normalize_features(predInputDf, contCols, modelBundle.normalization_params);

        predInputNormDf.info();
        self._export_df_for_debug("predInputNormDf", predInputNormDf);
        return predInputNormDf;
    # =================================================================================#
    def _export_df_for_debug(self, name, df):
        timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(fr"debug\{name}-{timeStamp}.csv");
