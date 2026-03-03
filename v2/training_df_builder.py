import logging
import pandas as pd

from abstractions.event_df_builder_base import EventDfBuilderBase
from abstractions.feature_builder_base import FeatureBuilderBase
from abstractions.item_id_builder_base import ItemIdBuilderBase
from abstractions.sample_builder_base import SampleBuilderBase
from abstractions.df_filter_base import DfFilterBase
from services.dataframe_debug_service import DataFrameDebugExportService
from models.datasource_paths_config import DataSourcePathsConfig
from abstractions.target_column_builder_base import  TargetColumnBuilderBase


#======================================================#
class TrainingDataBuilder:
    """
    Orchestrates the full training DataFrame construction pipeline.
    Builds purchase events, applies target labeling, runs sampling,
    applies filters, and executes the feature pipeline.
    """

    eventsDf: pd.DataFrame
    targetColumnBuilder: TargetColumnBuilderBase
    trainingPaths: dict[str, str]
    eventsDfBuilders: list[EventDfBuilderBase]
    sampleBuilders: list[SampleBuilderBase]
    featureBuilders: list[FeatureBuilderBase]
    sampleFilters: list[DfFilterBase]
    logger: logging.Logger

    #======================================================#
    def __init__(
            self,
            dfDebugExportService: DataFrameDebugExportService,
            eventsDfBuilders: list[EventDfBuilderBase],
            targetColumnBuilder: TargetColumnBuilderBase,
            itemIdBuilder: ItemIdBuilderBase,
            sampleBuilders: list[SampleBuilderBase],
            featureBuilders: list[FeatureBuilderBase],
            sampleFilters: list[DfFilterBase],
            dataSourcePathConfig: DataSourcePathsConfig
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
        self.itemIdBuilder = itemIdBuilder;
        self.logger = logging.getLogger(self.__class__.__name__)
        self.trainingPaths = dataSourcePathConfig.trainingPaths
        self.eventsDf = None
        self.eventsDfBuilders = eventsDfBuilders
        self.targetColumnBuilder = targetColumnBuilder
        self.sampleBuilders = sampleBuilders
        self.featureBuilders = featureBuilders
        self.sampleFilters = sampleFilters
        self.dfDebugExport =  dfDebugExportService;


    #======================================================#
    def build_df(self) -> pd.DataFrame:
        """
        Build the full training DataFrame by running the complete pipeline.

        :returns: Fully featured training DataFrame.
        :rtype: pd.DataFrame
        """
        self.logger.info("build_df(): start")

        df = self._build_events_df();
        df = self.targetColumnBuilder.build(df)
        df = self.itemIdBuilder.build(df);
        df = self._build_negative_samples(df)
        self.dfDebugExport.export(df, "training-afterNegatives");
        df = self._apply_feature_pipeline(df)
        self.dfDebugExport.export(df, "training-afterFeatures");
        df = self._build_sample_filters(df)
        self.dfDebugExport.export(df, "training-sampleFilters");
        self.logger.info("build_df(): done rows=%s cols=%s", len(df), len(df.columns))
        return df

    # ======================================================#
    def _build_events_df(self) -> pd.DataFrame:
        self.logger.info("Building Purchase event df.")
        eventDfs: list[pd.DataFrame] = []
        for builder in self.eventsDfBuilders:
            self.logger.info("running EventDfBuilder=%s", builder.__class__.__name__)

            df: pd.DataFrame = builder.build_df(self.trainingPaths)
            if df is not None and not df.empty:
                eventDfs.append(df)

        if len(eventDfs) == 0:
            self.logger.error("eventDfBuilders produced no events")
            raise RuntimeError("No event data produced by eventDfBuilders")

        return pd.concat(eventDfs, ignore_index=True)
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
            df = builder.filter(df)

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
            df = builder.build_samples(df)

        self.logger.info("_build_negative_samples(): done rows=%s", len(df))
        return df

    #======================================================#
    def _apply_feature_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all registered feature builders in dependency-resolved order.
        """

        self.logger.info(
            "_apply_feature_pipeline(): start builders=%s",
            len(self.featureBuilders)
        )

        # Resolve execution order based on column dependencies
        orderedBuilders = self.resolve_builder_order(self.featureBuilders, set(df.columns))

        for builder in orderedBuilders:
            builderName = builder.__class__.__name__
            self.logger.info("applying featureBuilder=%s", builderName)
            df = builder.build(df)

        self.logger.info(
            "_apply_feature_pipeline(): done rows=%s cols=%s",
            len(df),
            len(df.columns)
        )

        return df
    # ======================================================#

    def resolve_builder_order(self, builders: list[FeatureBuilderBase], initialColumns: set[str]) -> list[FeatureBuilderBase]:
        """
        Resolve a valid execution order for feature builders based on
        declared input/output column dependencies.

        :param builders: Feature builders to order.
        :param initialColumns: Columns initially present in the DataFrame.
        :returns: Builders in executable order.
        :raises RuntimeError: If dependencies cannot be resolved.
        """
        pending = list(builders)
        availableCols = set(initialColumns)
        ordered: list[FeatureBuilderBase] = []

        while pending:
            progress = False

            for builder in list(pending):
                required = set(builder.get_feature_names_in())

                if required.issubset(availableCols):
                    ordered.append(builder)
                    availableCols.update(builder.get_feature_names_out())
                    pending.remove(builder)
                    progress = True

            if not progress:
                missingInfo = {
                    b.__class__.__name__: list(
                        set(b.get_feature_names_in()) - availableCols
                    )
                    for b in pending
                }
                raise RuntimeError(
                    f"Feature dependency resolution failed: {missingInfo}"
                )

        return ordered
    # ======================================================#