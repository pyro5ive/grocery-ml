import logging
import pandas as pd
from abstractions.sample_builder_base import SampleBuilderBase


#======================================================#
class SameTripNegativeSampleBuilder(SampleBuilderBase):
    """
    Builds negative samples for trip days by expanding each item across all
    trip dates after its first purchase, filling missing combinations as non-purchase rows.
    Only generates negative samples after the item's activation date (first purchase).
    """

    didBuyTargetColName: str = "didBuy_target"
    itemIdColName: str = "itemId"
    itemNameColName: str = "item"
    dateColName: str = "date"
    sourceColName: str = "source"
    sourceColValue: str = "_same_trip_neg_sample_"

    logger: logging.Logger

    #======================================================#
    def __init__(self):
        """
        Initializes the sample builder.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("SameTripNegativeSampleBuilder initialized")

    #======================================================#
    def build_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Insert negative samples for trip days after each item's first purchase date.

        :param df: Input DataFrame containing positive purchase samples.
        :type df: pd.DataFrame
        :returns: Expanded DataFrame with same-trip negative samples inserted.
        :rtype: pd.DataFrame
        """
        self.logger.info("build_samples(): start rows=%s", len(df))
        df = self._insert_negative_samples(df)
        self.logger.info("build_samples(): done rows=%s", len(df))
        return df

    #======================================================#
    def _insert_negative_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build valid item/date combinations after each item's activation date
        and merge with existing data, filling missing rows as negative samples.

        :param df: Input DataFrame containing positive purchase samples.
        :type df: pd.DataFrame
        :returns: Expanded DataFrame with negative samples filled in.
        :rtype: pd.DataFrame
        """
        self.logger.info("_insert_negative_samples(): start rows=%s", len(df))

        df = df.copy()

        itemLookup: pd.DataFrame = (
            df[[self.itemIdColName, self.itemNameColName]]
            .drop_duplicates(subset=[self.itemIdColName])
        )

        firstPurchase: pd.Series = (
            df[df[self.didBuyTargetColName] == 1]
            .groupby(self.itemIdColName)[self.dateColName]
            .min()
        )

        rows: list = []
        allDates: list = df[self.dateColName].unique()

        for itemId, firstDate in firstPurchase.items():
            validDates = allDates[allDates >= firstDate]
            for d in validDates:
                rows.append({self.dateColName: d, self.itemIdColName: itemId})

        fullDf: pd.DataFrame = pd.DataFrame(rows)
        mergedDf: pd.DataFrame = fullDf.merge(df, on=[self.dateColName, self.itemIdColName], how="left")

        mergedDf[self.didBuyTargetColName] = mergedDf[self.didBuyTargetColName].fillna(False).astype(bool)

        mergedDf = mergedDf.merge(itemLookup, on=self.itemIdColName, how="left", suffixes=("", "_lookup"))
        mergedDf[self.itemNameColName] = mergedDf[self.itemNameColName].fillna(mergedDf[f"{self.itemNameColName}_lookup"])
        mergedDf = mergedDf.drop(columns=[f"{self.itemNameColName}_lookup"])

        mergedDf[self.sourceColName] = mergedDf[self.sourceColName].fillna(self.sourceColValue).astype(str)

        self.logger.info("_insert_negative_samples(): done rows=%s", len(mergedDf))
        return mergedDf