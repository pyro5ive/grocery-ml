import logging
from datetime import datetime
from training_df_builder import TrainingDataBuilder
from feature_normalizer.continous_feature_normalizer import ContinousFeatureNormalizer
from feature_schema import FeatureSchema
from model_builder.keras_model_builder import KerasModelBuilder
from prediction_service import PredictionService
from model_repo.model_bundle import  ModelBundle
from model_repo.keras_file_system_model_bundle_repo import KerasFileSystemModelRepository

class ExperimentRunner:
    trainingSources = {
        "walmart": r"..\data\training\walmart",
        "winndixie": r"..\data\training\winndixie\txt",
        "winndixieAdditional": r"..\data\training\winndixie\additionalTxtRcpts",
        "weather": r"..\data\weather\VisualCrossing-70062 2000-01-01 to 2026-23-1.csv"
    }
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__);
        self.trainingDfBuilder = TrainingDataBuilder(self.trainingSources);
        self.continuousFeatureNormalizer = ContinousFeatureNormalizer();
        self.featureSchema = FeatureSchema();
        self.kerasModelBuilder = KerasModelBuilder();
        self.modelBundle: ModelBundle = None;
        # self.normParams = None;
        self.trainingDf = None;
        self.trainingDfNorm = None;
        self.modelRepo = KerasFileSystemModelRepository();
        self.predictionService = None;
        # self.predictionInputDfBuilder = PredictionInputDfBuilder(self.liveSources, self.trainingSources, datetime(2026, 2, 8))
    ###########################################################################
    
    def run(self, buildParams, trainingParams, expDir):
        # df = self.predictionInputDfBuilder.build_df();
        # self._export_df_for_debug(df);

        # build normalized training df
        trainingDf = self.trainingDfBuilder.build_df();
        continuousCols = self.featureSchema.get_continuous_cols(trainingDf);
        normParams = self.continuousFeatureNormalizer.fit_normalization_params(continuousCols, trainingDf);
        trainingDfNorm  = self.continuousFeatureNormalizer.normalize_features( trainingDf, continuousCols, normParams );
        trainingDfNorm.info();
        #
        featCols = self.featureSchema.get_feature_cols(trainingDfNorm);
        targetCol = self.featureSchema.get_target_col(trainingDfNorm);
        # build/train model.
        model, modelTrainingHistory = self.kerasModelBuilder.build_and_train_model(trainingDfNorm, featCols, buildParams, trainingParams, targetCol);
        itemMappingDf = (trainingDf[['item', 'itemId']].drop_duplicates().copy())
        # Create Bundle

        modelBundle = ModelBundle(model, itemMappingDf, trainingDfNorm, normParams, modelTrainingHistory, buildParams, trainingParams);
        # Save bundle
        self.modelRepo.save(modelBundle, expDir);

        ## prediciton
        testPredDate = datetime.now();
        testPredictionService = PredictionService();
        predictionsResultDf = testPredictionService.run_prediction(modelBundle, testPredDate);
        predictionsResultDf.to_csv(expDir + "/predictions.csv");

        print(model.summary());
    ###########################################################################


    def _export_df_for_debug(self,df):
        timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(fr"debug\df-{timeStamp}.csv");
    ###########################################################################
        