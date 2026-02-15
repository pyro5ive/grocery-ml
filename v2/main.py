from datetime import datetime
import os
import punq


## Abstractions first
from abstractions.services.item_id_index_service_base import ItemIndexBuilderServiceBase
from abstractions.feature_builder_base import FeatureBuilderBase
from feature_builders.avg_days_between_item_purchases_feature_builder import AvgDaysBetweenItemPurchasesFeatureBuilder
from feature_builders.avg_days_between_trips_feature_builder import AvgDaysBetweenTripsFeatureBuilder
from feature_builders.days_since_last_purchase_feat_builder import DaysSinceLastPurchaseFeatureBuilder
from feature_builders.days_since_last_trip_feature_builder import DaysSinceLastTripFeatureBuilder
from feature_builders.expected_gap_ewma_feature_builder import ExpectedGapEwmaFeatureBuilder
from feature_builders.is_dst_feature_builder import IsDstFeatureBuilder
# from abstractions.services.item_id_index_service_base import ItemIndexBuilderServiceBase

## impl

from feature_builders.item_id_feature_builder import ItemIdFeatureBuilder
from experiment_runner import ExperimentRunner
from models.models import *

container = punq.Container();


container.register(FeatureBuilderBase,ItemIdFeatureBuilder,itemNameColName="item",itemIdColName="itemId")
container.register(FeatureBuilderBase, AvgDaysBetweenTripsFeatureBuilder)
container.register(FeatureBuilderBase, DaysSinceLastPurchaseFeatureBuilder)
container.register(FeatureBuilderBase, DaysSinceLastTripFeatureBuilder)
container.register(FeatureBuilderBase, ExpectedGapEwmaFeatureBuilder)
container.register(FeatureBuilderBase, IsDstFeatureBuilder)
container.register(FeatureBuilderBase, AvgDaysBetweenItemPurchasesFeatureBuilder)

expRunner = ExperimentRunner();

layers_cfg = [
    LayerSpec(units=8, activation="relu")
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
