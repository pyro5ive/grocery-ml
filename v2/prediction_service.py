import logging
import pandas as pd
import os
from prediction_input_df_builder import PredictionInputDfBuilder
from feature_normalizer.continous_feature_normalizer import ContinousFeatureNormalizer
from feature_schema import FeatureSchema
from datetime import datetime

class PredictionService:

    def __init__(this):
        this.logger = logging.getLogger(this.__class__.__name__);
        this.predictionInputEventsDfBuilder = PredictionInputDfBuilder(this.liveSources, this.trainingSources);
        this.continuousFeatureNormalizer = ContinousFeatureNormalizer();
        this.featureSchema = FeatureSchema();
        this.predInputDf = None;
    #=================================================================================#

    def run_prediction(this, kerasModelBundle, predictionDate: datetime):

        this.predInputDf = this.predictionInputEventsDfBuilder.build_df()
        this._export_df_for_debug(this.predInputDf);
        contCols = this.featureSchema.get_continuous_cols(this.trainingDf);
        this.continuousFeatureNormalizer.normalize_features(contCols, this.trainingDf, );
        this.trainingDf.info();
        this._export_df_for_debug();
        featCols = this.featureSchema.get_feature_cols(this.trainingDfNorm);
        targetCol = this.featureSchema.get_target_col(this.trainingDfNorm);

        this.predInputDf["prediction_df"] = this.predInputDf["prediction_df"][mask].reset_index(drop=True)
        this.predInputDf["x_item_idx"]     = pred_input["x_item_idx"][mask]
        pred_input["x_features"]     = pred_input["x_features"][mask]

        logger.info(f"kept {mask.sum()} rows out of {mask.size}")

        logger.info("Running Model.Predict()")
        predictions = model.predict([pred_input["x_features"], pred_input["x_item_idx"]])

        this.modelArtifacts = ModelArtifacts(model, this.trainingDfNorm, normParams, history, buildParams, trainingParams);
        print(model.summary());

        def run(this, buildParams, trainingParams):
            this.predictionDf = this.continuousFeatureNormalizer.normalize_features(this.trainingDf);
            normParams = this.continuousFeatureNormalizer.get_params();

    def _normalize(this):
        pass;

    def _predict(this):
        pass;
        this.model.predict();

    # =================================================================================#
    def _export_df_for_debug(this, df):
        timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(fr"debug\df-{timeStamp}.csv");
