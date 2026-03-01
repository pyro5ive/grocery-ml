import logging
from datetime import datetime

from abstractions.model_builder_base import ModelBuilderBase
from abstractions.normalizer_base import NormalizerBase
from feature_schema import FeatureSchema
from prediction_service import PredictionService
from models.model_bundle import ModelBundle
from abstractions.repos.model_bundle_repository_base import  ModelBundleRepositoryBase
from training_df_builder import TrainingDataBuilder


#======================================================#
class ExperimentRunner:
    """
    Orchestrates the full ML experiment lifecycle including training DataFrame
    construction, feature normalization, model training, bundle persistence,
    and prediction execution.
    """

    trainingSources: dict = {
        "walmart":              r"..\data\training\walmart",
        "winndixie":            r"..\data\training\winndixie\txt",
        "winndixieAdditional":  r"..\data\training\winndixie\additionalTxtRcpts",
        "weather":              r"..\data\weather\VisualCrossing-70062 2000-01-01 to 2026-23-1.csv"
    }

    trainingDfBuilder: TrainingDataBuilder
    normalizer: NormalizerBase
    featureSchema: FeatureSchema
    kerasModelBuilder: ModelBuilderBase
    modelRepo: ModelBundleRepositoryBase
    predictionService: PredictionService
    modelBundle: ModelBundle
    logger: logging.Logger

    #======================================================#
    def __init__(
        self,
        trainingDfBuilder: TrainingDataBuilder,
        normalizer: NormalizerBase,
        featureSchema: FeatureSchema,
        kerasModelBuilder: ModelBuilderBase,
        modelRepo: ModelBundleRepositoryBase,
        predictionService: PredictionService
    ):
        """
        :param trainingDfBuilder: Builds the full training DataFrame from all sources.
        :type trainingDfBuilder: TrainingDataBuilder
        :param normalizer: Normalizes continuous feature columns.
        :type normalizer: NormalizerBase
        :param featureSchema: Resolves feature and target column names from the DataFrame.
        :type featureSchema: FeatureSchema
        :param kerasModelBuilder: Builds and trains the Keras model.
        :type kerasModelBuilder: KerasModelBuilder
        :param modelRepo: Persists and loads model bundles from the filesystem.
        :type modelRepo: KerasFileSystemModelRepository
        :param predictionService: Runs predictions against a trained model bundle.
        :type predictionService: PredictionService
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.trainingDfBuilder = trainingDfBuilder
        self.normalizer = normalizer
        self.featureSchema = featureSchema
        self.kerasModelBuilder = kerasModelBuilder
        self.modelRepo = modelRepo
        self.predictionService = predictionService
        self.modelBundle = None
        self.logger.info("ExperimentRunner initialized")

    #======================================================#
    def run(self, buildParams: dict, trainingParams: dict, expDir: str) -> None:
        """
        Execute the full experiment lifecycle: build training data, normalize features,
        train model, persist bundle, and run prediction.

        :param buildParams: Model architecture parameters.
        :type buildParams: dict
        :param trainingParams: Model training parameters.
        :type trainingParams: dict
        :param expDir: Directory path to save the model bundle and prediction output.
        :type expDir: str
        """
        self.logger.info("run(): start expDir=%s", expDir)

        trainingDf = self.trainingDfBuilder.build_df()

        continuousCols: list = self.featureSchema.get_continuous_cols(trainingDf)
        trainingDfNorm = self.normalizer.fit_transform(continuousCols, trainingDf)
        trainingDfNorm.info()
        featCols: list = self.featureSchema.get_feature_cols(trainingDfNorm)
        targetCol: str = self.featureSchema.get_target_col(trainingDfNorm)

        model, modelTrainingHistory = self.kerasModelBuilder.build_and_train_model(
            trainingDfNorm, featCols, buildParams, trainingParams, targetCol
        )

        itemMappingDf = trainingDf[["item", "itemId"]].drop_duplicates().copy()

        modelBundle: ModelBundle = ModelBundle(
            model,
            trainingDfNorm,
            itemMappingDf,
            self.normalizer.get_params(),
            modelTrainingHistory,
            buildParams,
            trainingParams
        )

        self.modelRepo.save(modelBundle, expDir)

        # testPredDate: datetime = datetime.now()
        # predictionsResultDf = self.predictionService.run_prediction(modelBundle, testPredDate)
        # predictionsResultDf.to_csv(expDir + "/predictions.csv")

        print(model.summary())
        self.logger.info("run(): done expDir=%s", expDir)

    #======================================================#
    def _export_df_for_debug(self, df) -> None:
        """
        Export the DataFrame to a timestamped CSV file for debugging purposes.

        :param df: DataFrame to export.
        :type df: pd.DataFrame
        """
        timeStamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(fr"debug\df-{timeStamp}.csv")
        self.logger.info("_export_df_for_debug(): exported timeStamp=%s", timeStamp)