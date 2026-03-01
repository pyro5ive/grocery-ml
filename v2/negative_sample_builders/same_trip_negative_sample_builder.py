import logging
import pandas as pd
from abstractions.sample_builder_base import SampleBuilderBase


#======================================================#
class SameTripNegativeSampleBuilder(SampleBuilderBase):

    didBuyTargetColName: str = "didBuy_target"
    itemIdColName: str = "itemId"
    itemNameColName: str = "item"
    dateColName: str = "date"
    sourceColName: str = "source"
    sourceColValue: str = "_same_trip_neg_sample_"

    logger: logging.Logger

    #======================================================#
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("SameTripNegativeSampleBuilder initialized")

    #======================================================#
    def build_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("build_samples(): start rows=%s", len(df))
        df = self._insert_negative_samples(df)
        self.logger.info("build_samples(): done rows=%s", len(df))
        return df

    #======================================================#
    def _insert_negative_samples(self, df: pd.DataFrame) -> pd.DataFrame:
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

        mergedDf: pd.DataFrame = fullDf.merge(
            df,
            on=[self.dateColName, self.itemIdColName],
            how="left"
        )

        mergedDf[self.didBuyTargetColName] = (
            mergedDf[self.didBuyTargetColName]
            .fillna(False)
            .astype(bool)
        )

        mergedDf = mergedDf.merge(
            itemLookup,
            on=self.itemIdColName,
            how="left",
            suffixes=("", "_lookup")
        )

        mergedDf[self.itemNameColName] = mergedDf[self.itemNameColName].fillna(
            mergedDf[f"{self.itemNameColName}_lookup"]
        )

        mergedDf = mergedDf.drop(columns=[f"{self.itemNameColName}_lookup"])

        mergedDf[self.sourceColName] = (
            mergedDf[self.sourceColName]
            .fillna(self.sourceColValue)
            .astype(str)
        )

        mergedDf = self._reorder_columns(mergedDf)

        self.logger.info("_insert_negative_samples(): done rows=%s", len(mergedDf))
        return mergedDf

    #======================================================#
    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce stable column order:
        index 3 -> item
        index 4 -> itemId
        """
        cols = list(df.columns)

        if self.itemNameColName not in cols:
            raise ValueError(f"missing column '{self.itemNameColName}'")

        if self.itemIdColName not in cols:
            raise ValueError(f"missing column '{self.itemIdColName}'")

        cols.remove(self.itemNameColName)
        cols.remove(self.itemIdColName)

        cols.insert(3, self.itemNameColName)
        cols.insert(4, self.itemIdColName)

        return df.reindex(columns=cols)