import logging
import pandas as pd
import numpy as np

from abstractions.normalizer_base import NormalizerBase
from abstractions.prediction_feature_builder_base import PredictionFeatureBuilderBase
from models.model_bundle import ModelBundle
from prediction_input_df_builder import PredictionInputDfBuilder
# from feature_normalizer.continous_feature_normalizer import ContinousFeatureNormalizer

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

    def __init__(
            self,
            predictionInputDfBuilder: PredictionInputDfBuilder,
            normalizer: NormalizerBase,


    ):
        self.logger = logging.getLogger(self.__class__.__name__);
        self.predictionInputEventsDfBuilder = predictionInputDfBuilder;
        self.continuousFeatureNormalizer = normalizer;
        self.featureSchema = FeatureSchema();
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

        # known_ids = set(combined_df_frozen["itemId"].unique())
        # mask = pred_input["prediction_df"]["itemId"].isin(known_ids)
        #
        # pred_input["prediction_df"] = pred_input["prediction_df"][mask].reset_index(drop=True)
        # pred_input["x_item_idx"] = pred_input["x_item_idx"][mask]
        # pred_input["x_features"] = pred_input["x_features"][mask]
        #
        # logger.info(f"kept {mask.sum()} rows out of {mask.size}")

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
