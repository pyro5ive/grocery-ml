import logging
import pandas as pd
from abstractions.sample_builder_base import SampleBuilderBase


#======================================================#
class NonTripNegativeSampleBuilder(SampleBuilderBase):
    """
    Builds negative samples for non-trip days by creating a full calendar
    of item/date combinations and filling missing days as non-purchase rows.
    Covers a rolling window of days back from the most recent date in the DataFrame.
    """

    itemIdColName: str = "itemId"
    itemNameColName: str = "item"
    dateColName: str = "date"
    didBuyTargetColName: str = "didBuy_target"
    sourceColName: str = "source"
    sourceColValue: str = "_neg_sample_no_trip"
    windowDays: int = 365

    logger: logging.Logger
    #======================================================#
    def __init__(self):
        """
        Initializes the sample builder.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        # self.logger.info("NonTripNegativeSampleBuilder initialized")

    #======================================================#
    def build_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Insert negative samples for non-trip days over a rolling window.

        :param df: Input DataFrame containing positive purchase samples.
        :type df: pd.DataFrame
        :returns: Expanded DataFrame with non-trip negative samples inserted.
        :rtype: pd.DataFrame
        """
        self.logger.info("build_samples(): start rows=%s", len(df))
        df = self._create(df)
        self.logger.info("build_samples(): done rows=%s", len(df))
        return df

    #======================================================#
    def build_items(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract a deduplicated itemId to item name lookup from the DataFrame.

        :param df: Input DataFrame containing itemId and item name columns.
        :type df: pd.DataFrame
        :returns: DataFrame with one row per unique itemId and its item name.
        :rtype: pd.DataFrame
        """
        itemLookup: pd.DataFrame = (
            df[[self.itemIdColName, self.itemNameColName]]
            .drop_duplicates(subset=[self.itemIdColName])
        )
        return itemLookup

    #======================================================#
    def _create(self, df: pd.DataFrame) -> pd.DataFrame:
        originalColumns: list[str] = list(df.columns)

        df = df.copy()
        df[self.dateColName] = pd.to_datetime(df[self.dateColName]).dt.normalize()

        itemLookup: pd.DataFrame = df[[self.itemIdColName, self.itemNameColName]].drop_duplicates(self.itemIdColName)

        maxDate: pd.Timestamp = df[self.dateColName].max()
        minDate: pd.Timestamp = maxDate - pd.Timedelta(days=self.windowDays - 1)

        calendar: pd.DataFrame = (
            itemLookup[[self.itemIdColName]]
            .merge(pd.DataFrame({self.dateColName: pd.date_range(minDate, maxDate, freq="D")}), how="cross")
        )

        merged: pd.DataFrame = calendar.merge(df, on=[self.itemIdColName, self.dateColName], how="left")

        merged[self.didBuyTargetColName] = merged[self.didBuyTargetColName].fillna(False).astype(bool)
        merged[self.sourceColName] = merged[self.sourceColName].fillna(self.sourceColValue)

        merged = merged.merge(itemLookup, on=self.itemIdColName, how="left", suffixes=("", "_lk"))
        merged[self.itemNameColName] = merged[self.itemNameColName].fillna(merged[f"{self.itemNameColName}_lk"])
        merged = merged.drop(columns=[f"{self.itemNameColName}_lk"])

        merged = merged.sort_values([self.itemIdColName, self.dateColName]).reset_index(drop=True)

        return merged[originalColumns]
    #--------------------------#