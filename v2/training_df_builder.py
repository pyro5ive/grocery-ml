import logging
import pandas as pd
from abstractions.feature_builder_base import FeatureBuilderBase
from abstractions.sample_builder_base import SampleBuilderBase
from abstractions.df_filter_base import DfFilterBase
from feature_builders.item_id_feature_builder import ItemIdFeatureBuilder
from purchase_event_builders.purchase_event_aggregate_builder import PurchaseEventAggregateBuilder


#======================================================#
class TrainingDataBuilder:
    """
    Orchestrates the full training DataFrame construction pipeline.
    Builds purchase events, applies negative sampling, runs the feature
    pipeline, and returns a fully featured training DataFrame.
    """

    purchaseEventAggregateBuilder: PurchaseEventAggregateBuilder
    itemIdFeatureBuilder: ItemIdFeatureBuilder
    sameTripNegativeSampleBuilder: SampleBuilderBase
    nonTripNegativeSampleBuilder: SampleBuilderBase
    sameTripQtyCombiner: DfFilterBase
    featureBuilders: list[FeatureBuilderBase]
    logger: logging.Logger

    #======================================================#
    def __init__(
        self,
        purchaseEventAggregateBuilder: PurchaseEventAggregateBuilder,
        itemIdFeatureBuilder: ItemIdFeatureBuilder,
        sameTripNegativeSampleBuilder: SampleBuilderBase,
        nonTripNegativeSampleBuilder: SampleBuilderBase,
        sameTripQtyCombiner: DfFilterBase,
        featureBuilders: list[FeatureBuilderBase]
    ):
        """
        :param purchaseEventAggregateBuilder: Builds the raw purchase events DataFrame from all sources.
        :type purchaseEventAggregateBuilder: PurchaseEventAggregateBuilder
        :param itemIdFeatureBuilder: Builds the itemId feature column using the item index service.
        :type itemIdFeatureBuilder: ItemIdFeatureBuilder
        :param sameTripNegativeSampleBuilder: Inserts negative samples for same-trip days.
        :type sameTripNegativeSampleBuilder: SampleBuilderBase
        :param nonTripNegativeSampleBuilder: Inserts negative samples for non-trip days.
        :type nonTripNegativeSampleBuilder: SampleBuilderBase
        :param sameTripQtyCombiner: Combines duplicate date/itemId rows by summing qty.
        :type sameTripQtyCombiner: DfFilterBase
        :param featureBuilders: Ordered list of feature builders to apply in the pipeline.
        :type featureBuilders: list[FeatureBuilderBase]
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.purchaseEventAggregateBuilder = purchaseEventAggregateBuilder
        self.itemIdFeatureBuilder = itemIdFeatureBuilder
        self.sameTripNegativeSampleBuilder = sameTripNegativeSampleBuilder
        self.nonTripNegativeSampleBuilder = nonTripNegativeSampleBuilder
        self.sameTripQtyCombiner = sameTripQtyCombiner
        self.featureBuilders = featureBuilders
        self.logger.info("TrainingDataBuilder initialized featureBuilders=%s", len(self.featureBuilders))

    #======================================================#
    def build_df(self) -> pd.DataFrame:
        """
        Build the full training DataFrame by running the complete pipeline.

        :returns: Fully featured training DataFrame.
        :rtype: pd.DataFrame
        """
        self.logger.info("build_df(): start")

        df: pd.DataFrame = self.purchaseEventAggregateBuilder.build_df()
        df = self._build_target_col(df)
        df = self.itemIdFeatureBuilder.build(df)
        df = self._build_negative_samples(df)
        df = self.sameTripQtyCombiner.filter(df)
        df = self._apply_feature_pipeline(df)

        self.logger.info("build_df(): done rows=%s cols=%s", len(df), len(df.columns))
        return df

    #======================================================#
    def _build_target_col(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add the didBuy_target boolean column, defaulting all rows to True.
        Negative sample builders will set their rows to False after this step.

        :param df: Input DataFrame of purchase events.
        :type df: pd.DataFrame
        :returns: DataFrame with didBuy_target column added.
        :rtype: pd.DataFrame
        """
        self.logger.info("_build_target_col(): start")
        df["didBuy_target"] = True
        df["didBuy_target"] = df["didBuy_target"].astype(bool)
        return df

    #======================================================#
    def _build_negative_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply same-trip and non-trip negative sample builders to the DataFrame.

        :param df: Input DataFrame containing positive purchase rows.
        :type df: pd.DataFrame
        :returns: Expanded DataFrame with negative samples inserted.
        :rtype: pd.DataFrame
        """
        self.logger.info("_build_negative_samples(): start rows=%s", len(df))
        df = self.sameTripNegativeSampleBuilder.build_samples(df)
        df = self.nonTripNegativeSampleBuilder.build_samples(df)
        self.logger.info("_build_negative_samples(): done rows=%s", len(df))
        return df

    #======================================================#
    def _apply_feature_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all registered feature builders sequentially to the DataFrame.

        :param df: Input DataFrame to run through the feature pipeline.
        :type df: pd.DataFrame
        :returns: DataFrame with all feature columns added.
        :rtype: pd.DataFrame
        """
        self.logger.info("_apply_feature_pipeline(): start builders=%s", len(self.featureBuilders))

        for builder in self.featureBuilders:
            builderName: str = builder.__class__.__name__
            self.logger.info("_apply_feature_pipeline(): applying builder=%s", builderName)
            df = builder.build(df)

        self.logger.info("_apply_feature_pipeline(): done rows=%s cols=%s", len(df), len(df.columns))
        return df