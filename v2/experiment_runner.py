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

    liveSources = {
        "walmart": r"..\data\live\walmart",
        "winndixie": r"..\data\live\winndixie\txt",
        "winndixieAdditional": r"..\data\live\winndixie\additionalTxtRcpts",
        "weather": r"..\date\weather\VisualCrossing-70062 2000-01-01 to 2026-23-1.csv"
    }
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__);
        self.trainingDfBuilder = TrainingDataBuilder(self.trainingSources);
        self.continuousFeatureNormalizer = ContinousFeatureNormalizer();
        self.featureSchema = FeatureSchema();
        self.kerasModelBuilder = KerasModelBuilder();
        self.modelBundle: ModelBundle = None;
        self.normParams = None;
        self.trainingDf = None;
        self.trainingDfNorm = None;
        self.modelRepo = KerasFileSystemModelRepository();
        self.predictionService = None;
        # self.predictionInputDfBuilder = PredictionInputDfBuilder(self.liveSources, self.trainingSources, datetime(2026, 2, 8))
    ###########################################################################
    
    def run(self, buildParams, trainingParams, baseDir):

        # df = self.predictionInputDfBuilder.build_df();
        # self._export_df_for_debug(df);

        self.trainingDf = self.trainingDfBuilder.build_df();
        contCols = self.featureSchema.get_continuous_cols(self.trainingDf);
        self.normParams = self.continuousFeatureNormalizer.fit_normalization_params( contCols, self.trainingDf,);
        self.trainingDfNorm  = self.continuousFeatureNormalizer.normalize_features(self.trainingDf, self.normParams);
        self.trainingDf.info();

        featCols = self.featureSchema.get_feature_cols(self.trainingDfNorm);
        targetCol = self.featureSchema.get_target_col(self.trainingDfNorm);
        model, history = self.kerasModelBuilder.build_and_train_model(self.trainingDfNorm, featCols, buildParams, trainingParams, targetCol);
        self.modelBundle = ModelBundle(model, self.trainingDfNorm, self.normParams, history, buildParams, trainingParams);
        self.modelRepo.save_all(self.modelBundle, baseDir);

        ## prediciton
        testPredDate = datetime.now();
        self.predictionService = PredictionService(testPredDate);

        print(model.summary());
    ###########################################################################


    def _export_df_for_debug(self,df):
        timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(fr"debug\df-{timeStamp}.csv");
    ###########################################################################
        