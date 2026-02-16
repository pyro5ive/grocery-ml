import logging
import pandas as pd
from abstractions.feature_builder_base import FeatureBuilderBase
from abstractions.sample_builder_base import SampleBuilderBase
from abstractions.df_filter_base import DfFilterBase
from purchase_event_builders.purchase_event_aggregate_builder import PurchaseEventAggregateBuilder
from abstractions.target_column_builder_base import  TargetColumnBuilderBase


#======================================================#
class TrainingDataBuilder:
    """
    Orchestrates the full training DataFrame construction pipeline.
    Builds purchase events, applies target labeling, runs sampling,
    applies filters, and executes the feature pipeline.
    """

    purchaseEventAggregateBuilder: PurchaseEventAggregateBuilder
    targetColumnBuilder: TargetColumnBuilderBase
    sampleBuilders: list[SampleBuilderBase]
    featureBuilders: list[FeatureBuilderBase]
    sampleFilters: list[DfFilterBase]
    logger: logging.Logger

    #======================================================#
    def __init__(
            self,
            purchaseEventAggregateBuilder: PurchaseEventAggregateBuilder,
            targetColumnBuilder: TargetColumnBuilderBase,
            sampleBuilders: list[SampleBuilderBase],
            featureBuilders: list[FeatureBuilderBase],
            sampleFilters: list[DfFilterBase]
    ):
        """
        Initialize the TrainingDataBuilder.

        :param purchaseEventAggregateBuilder: Aggregates purchase event DataFrames
            from all registered event builders into a single events DataFrame.
        :type purchaseEventAggregateBuilder: PurchaseEventAggregateBuilder

        :param targetColumnBuilder: Builds the training target/label column.
            This component is training-only.
        :type targetColumnBuilder: TargetColumnBuilder

        :param sampleBuilders: Ordered list of sample builders responsible for
            inserting negative samples into the training DataFrame.
        :type sampleBuilders: list[SampleBuilderBase]

        :param featureBuilders: Ordered list of feature builders that derive and
            add model feature columns to the DataFrame.
        :type featureBuilders: list[FeatureBuilderBase]

        :param sampleFilters: Ordered list of DataFrame filters applied after
            sampling to clean, merge, or reduce the training data.
        :type sampleFilters: list[DfFilterBase]
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.purchaseEventAggregateBuilder = purchaseEventAggregateBuilder
        self.targetColumnBuilder = targetColumnBuilder
        self.sampleBuilders = sampleBuilders
        self.featureBuilders = featureBuilders
        self.sampleFilters = sampleFilters

        self.logger.info(
            "TrainingDataBuilder initialized sampleBuilders=%s featureBuilders=%s sampleFilters=%s",
            len(self.sampleBuilders),
            len(self.featureBuilders),
            len(self.sampleFilters)
        )

    #======================================================#
    def build_df(self) -> pd.DataFrame:
        """
        Build the full training DataFrame by running the complete pipeline.

        :returns: Fully featured training DataFrame.
        :rtype: pd.DataFrame
        """
        self.logger.info("build_df(): start")

        df: pd.DataFrame = self.purchaseEventAggregateBuilder.build_df()

        df = self.targetColumnBuilder.build(df)
        df = self._apply_feature_pipeline(df)
        df = self._build_negative_samples(df)
        df = self._build_sample_filters(df)

        self.logger.info("build_df(): done rows=%s cols=%s", len(df), len(df.columns))
        return df

    #======================================================#
    def _build_sample_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all registered DataFrame filters sequentially.

        :param df: Input DataFrame to filter.
        :type df: pd.DataFrame
        :returns: Filtered DataFrame.
        :rtype: pd.DataFrame
        """
        self.logger.info("_build_sample_filters(): start rows=%s", len(df))

        for builder in self.sampleFilters:
            builderName: str = builder.__class__.__name__
            self.logger.info("applying filter=%s", builderName)
            df = builder.build(df)

        self.logger.info("_build_sample_filters(): done rows=%s", len(df))
        return df

    #======================================================#
    def _build_negative_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all registered sample builders sequentially.

        :param df: Input DataFrame containing positive purchase rows.
        :type df: pd.DataFrame
        :returns: Expanded DataFrame with negative samples inserted.
        :rtype: pd.DataFrame
        """
        self.logger.info("_build_negative_samples(): start rows=%s", len(df))

        for builder in self.sampleBuilders:
            builderName: str = builder.__class__.__name__
            self.logger.info("applying sampleBuilder=%s", builderName)
            df = builder.build(df)

        self.logger.info("_build_negative_samples(): done rows=%s", len(df))
        return df

    #======================================================#
    def _apply_feature_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all registered feature builders sequentially.

        :param df: Input DataFrame to run through the feature pipeline.
        :type df: pd.DataFrame
        :returns: DataFrame with all feature columns added.
        :rtype: pd.DataFrame
        """
        self.logger.info("_apply_feature_pipeline(): start builders=%s", len(self.featureBuilders))

        for builder in self.featureBuilders:
            builderName: str = builder.__class__.__name__
            self.logger.info("applying featureBuilder=%s", builderName)
            df = builder.build(df)

        self.logger.info("_apply_feature_pipeline(): done rows=%s cols=%s", len(df), len(df.columns))
        return df
