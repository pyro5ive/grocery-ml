import logging
import pandas as pd
from datetime import datetime
from purchase_event_builders.winn_dixie_events_df_builder import WinnDixieEventsDfBuilder
from training_df_builder import TrainingDataBuilder
from feature_normalizer.continous_feature_normalizer import ContinousFeatureNormalizer
from feature_schema import FeatureSchema
from model_builder.keras_model_builder import KerasModelBuilder
from model_repo.model_repo import ModelArtifacts
from model_repo.model_repo import KerasFileSystemModelRepository


class ExperimentRunner:

    trainingSources  = {
        "walmart": r"data\training\walmart",
        "winndixie": r"..\data\training\winndixie\txt",
        "winndixieAdditional" : r"..\data\training\winndixie\additionalTxtRcpts",
        "weather": r"data\weather\VisualCrossing-70062 2000-01-01 to 2026-23-1.csv"
    }
    
    liveSources  = {
        "walmart": r"data\live\walmart",
        "winndixie": r"data\live\winndixie\txt",
        "winndixieAdditional" : r"data\live\winndixie\additionalTxtRcpts",
        "weather": r"data\weather\VisualCrossing-70062 2000-01-01 to 2026-23-1.csv"
    }
    
    def __init__(this):
        this.logger = logging.getLogger(this.__class__.__name__);
        this.trainingDfBuilder = TrainingDataBuilder(this.trainingSources);
        this.continuousFeatureNormalizer = ContinousFeatureNormalizer();
        this.featureSchema = FeatureSchema();
        this.kerasModelBuilder = KerasModelBuilder();
        this.modelArtifacts: ModelArtifacts = None;
        this.modelRepo = KerasFileSystemModelRepository();
    ###########################################################################
    
    def run(this, buildParams, trainingParams):
        this.trainingDf = TrainingDataBuilder(this.trainingSources).build_df();
        contCols = this.featureSchema.get_continuous_cols(this.trainingDf);
        this.continuousFeatureNormalizer.fit_normalization_params( contCols, this.trainingDf,);
        this.trainingDf  = this.continuousFeatureNormalizer.normalize_features(this.trainingDf);
        this.trainingDf.info();
        this._export_df_for_debug();
        featCols = this.featureSchema.get_feature_cols(this.trainingDf);
        targetCol = this.featureSchema.get_target_col(this.trainingDf);
        model, history = this.kerasModelBuilder.build_and_train_model(this.trainingDf, featCols, buildParams, trainingParams, targetCol);

        normParams = this.continuousFeatureNormalizer.get_params();

        this.modelArtifacts = ModelArtifacts(model, this.trainingDf, normParams, history, buildParams, trainingParams);
        print(model.summary());
        print(this.modelArtifacts.build_params);

    ###########################################################################


    def _export_df_for_debug(this):
        timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        this.trainingDf.to_csv(fr"debug\trainingDf-{timeStamp}.csv");
    ###########################################################################
        