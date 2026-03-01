from datetime import datetime
import os

import pandas as pd
import punq

## Abstractions
from abstractions.df_filter_base import DfFilterBase
from abstractions.event_df_builder_base import EventDfBuilderBase
from abstractions.item_id_builder_base import ItemIdBuilderBase
from abstractions.model_builder_base import ModelBuilderBase
from abstractions.normalizer_base import NormalizerBase
from abstractions.prediction_feature_builder_base import PredictionFeatureBuilderBase
from abstractions.purchase_event_mapper_base import PurchaseEventMapperBase
from abstractions.repos.model_bundle_repository_base import ModelBundleRepositoryBase
from abstractions.sample_builder_base import SampleBuilderBase
from abstractions.feature_builder_base import FeatureBuilderBase
from abstractions.services.item_id_index_service_base import ItemIndexBuilderServiceBase
from abstractions.services.weather_service_base import WeatherServiceBase
from abstractions.target_column_builder_base import TargetColumnBuilderBase
from dataframe_debug_service import DataFrameDebugExportService
from feature_builders.avg_days_between_item_purchases_feature_builder import AvgDaysBetweenItemPurchasesFeatureBuilder
from feature_builders.avg_days_between_trips_feature_builder import AvgDaysBetweenTripsFeatureBuilder
from feature_builders.days_since_last_purchase_feat_builder import DaysSinceLastPurchaseFeatureBuilder
from feature_builders.days_since_last_trip_feature_builder import DaysSinceLastTripFeatureBuilder
from feature_builders.expected_gap_ewma_feature_builder import ExpectedGapEwmaFeatureBuilder
from feature_builders.is_dst_feature_builder import IsDstFeatureBuilder
# from abstractions.services.item_id_index_service_base import ItemIndexBuilderServiceBase


from experiment_runner import ExperimentRunner
from feature_builders.item_supply_level_feature_builder import ItemSupplyLevelFeatureBuilder
from feature_builders.item_total_purchase_count_feature_builder import ItemTotalPurchaseCountFeatureBuilder
from feature_builders.payday_prox_feature_builder import PaydayProximityFeatureBuilder
from feature_builders.school_schedule_feat_builder import SchoolScheduleFeatureBuilder
from feature_builders.weather_forecast_feature_builder import WeatherForecastFeatureBuilder
from feature_builders.weather_history_feat_builder import WeatherHistoryFeatureBuilder
from feature_normalizer.continous_feature_normalizer import ContinuousFeatureNormalizer
from feature_schema import FeatureSchema
from item_id.item_id_builder import ItemIdBuilder
from model_builder.keras_model_builder import KerasModelBuilder
from model_repo.keras_file_system_model_bundle_repo import KerasFileSystemModelBundleRepository
from models.models import *
from negative_sample_builders.non_trip_negative_sample_builder import NonTripNegativeSampleBuilder
from negative_sample_builders.same_trip_negative_sample_builder import SameTripNegativeSampleBuilder
from prediction_input_df_builder import PredictionInputDfBuilder
from prediction_service import PredictionService
from purchase_event_builders.event_df_builders.mappers.winn_dixie_events_df_mapper import WinnDixieReceiptToPurchaseEventMapper
from purchase_event_builders.event_df_builders.winn_dixie_recpt_parser import WinnDixieRecptParser
from purchase_event_builders.prediction_date_events_df_builder import PredictionDateEventsDfBuilder
from purchase_event_builders.event_df_builders.winn_dixie_events_df_builder import WinnDixieEventsDfBuilder
from sample_filters.combine_same_trip_qty import SameTripQtyCombiner
from sample_filters.rare_purchase_sample_filter import RarePurchaseFilter
from item_id.item_index_service.item_index_builder_service import ItemIndexBuilderService
from services.weather.nws_weather_service import NwsWeatherService
from target_col_builder.target_col_builder import TargetColumnBuilder
from training_df_builder import TrainingDataBuilder
from models.datasource_paths_config import DataSourcePathsConfig



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
    "weather": r"..\data\weather\VisualCrossing-70062 2000-01-01 to 2026-23-1.csv"
}

dataSourcePaths = DataSourcePathsConfig(trainingSources, liveSources);

serviceProvider = punq.Container();
serviceProvider.register(DataFrameDebugExportService);
serviceProvider.register(ExperimentRunner)
serviceProvider.register(TrainingDataBuilder);
serviceProvider.register(TargetColumnBuilderBase, TargetColumnBuilder, targetColName="didBuy_target");
serviceProvider.register(WeatherServiceBase, NwsWeatherService, userAgent="(grocery-ml, nolabizit@gmail.com)")
serviceProvider.register(FeatureSchema);
serviceProvider.register(ModelBuilderBase, KerasModelBuilder);
serviceProvider.register(ModelBundleRepositoryBase, KerasFileSystemModelBundleRepository);
serviceProvider.register(PredictionService);
serviceProvider.register(TrainingDataBuilder);
serviceProvider.register(PredictionInputDfBuilder);
# Normalizer
serviceProvider.register(NormalizerBase, ContinuousFeatureNormalizer)
#======================================================#
#Filter
serviceProvider.register(DfFilterBase, SameTripQtyCombiner)
serviceProvider.register(DfFilterBase, RarePurchaseFilter, minPurchaseThreshold=1)
#======================================================#
# Sample
serviceProvider.register(SampleBuilderBase, SameTripNegativeSampleBuilder)
serviceProvider.register(SampleBuilderBase, NonTripNegativeSampleBuilder)
#======================================================#
# Purchase Events
#  winndixie
serviceProvider.register(PurchaseEventMapperBase, WinnDixieReceiptToPurchaseEventMapper)
serviceProvider.register(WinnDixieRecptParser);
serviceProvider.register(EventDfBuilderBase, WinnDixieEventsDfBuilder)
#  winndixie json
#serviceProvider.register(EventDfBuilderBase, WinnDixieEventsFromJsonDfBuilder)
#serviceProvider.register(PurchaseEventMapperBase, WinnDixieJsonToPurchaseEventMapper)
# walmart
# serviceProvider.register(EventDfBuilderBase, WalMartEventsDfBuilder)
# serviceProvider.register(PurchaseEventMapperBase, WalMartReceiptToPurchaseEventMapper)
# manual
# serviceProvider.register(EventDfBuilderBase, ManualEntryEventsDfBuilder)
# serviceProvider.register(EventDfBuilderBase, ManualEntryEventsDfBuilder)
# source path config for event df builders
serviceProvider.register(DataSourcePathsConfig, instance=dataSourcePaths)
serviceProvider.register(PredictionDateEventsDfBuilder)
#======================================================#
# Feature Builders
#
# serviceProvider.register(
#     FeatureBuilderBase,
#     factory=lambda: ItemIdFeatureBuilder(
#         indexBuilder=serviceProvider.resolve(ItemIndexBuilderServiceFactory).create(),
#         itemNameColName="item",
#         itemIdColName="itemId"
#     )
# )
# serviceProvider.register(ItemIndexBuilderServiceFactory);
serviceProvider.register(ItemIndexBuilderServiceBase, ItemIndexBuilderService);
serviceProvider.register(ItemIdBuilderBase, ItemIdBuilder)
serviceProvider.register(FeatureBuilderBase, WeatherHistoryFeatureBuilder, sourcePath=r"..\data\weather\VisualCrossing-70062 2000-01-01 to 2026-23-1.csv")
serviceProvider.register(PredictionFeatureBuilderBase, WeatherForecastFeatureBuilder)
serviceProvider.register(FeatureBuilderBase, SchoolScheduleFeatureBuilder)
serviceProvider.register(FeatureBuilderBase, IsDstFeatureBuilder)
serviceProvider.register(FeatureBuilderBase, PaydayProximityFeatureBuilder, personName="ang", anchorPayday=pd.Timestamp("2026-01-23"))
serviceProvider.register(FeatureBuilderBase, PaydayProximityFeatureBuilder, personName="sjm", anchorPayday=pd.Timestamp("2026-01-30"))
serviceProvider.register(FeatureBuilderBase, DaysSinceLastPurchaseFeatureBuilder)
serviceProvider.register(FeatureBuilderBase, AvgDaysBetweenItemPurchasesFeatureBuilder)
serviceProvider.register(FeatureBuilderBase, ItemTotalPurchaseCountFeatureBuilder)
serviceProvider.register(FeatureBuilderBase, ItemSupplyLevelFeatureBuilder)
serviceProvider.register(FeatureBuilderBase, DaysSinceLastTripFeatureBuilder)
serviceProvider.register(FeatureBuilderBase, AvgDaysBetweenTripsFeatureBuilder)
serviceProvider.register(FeatureBuilderBase, ExpectedGapEwmaFeatureBuilder)


expRunner = serviceProvider.resolve(ExperimentRunner);

layers_cfg = [
    LayerSpec(units=1, activation="relu")
]

build_config = BuildParams(
    embeddingDimCount=4,
    layers=layers_cfg,
    outputActivation="sigmoid",
    optimizer="adam",
    learningRate=0.001,
    loss="binary_crossentropy",
    metrics=["AUC"]
)

train_config = TrainingParams(
    epochs=1,
    batchSize=4
)

timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("debug", f"test-exp_{timestamp}")

expRunner.run(build_config, train_config, run_dir);
