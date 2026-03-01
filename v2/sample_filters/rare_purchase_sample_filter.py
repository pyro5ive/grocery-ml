import logging
import pandas as pd
from abstractions.df_filter_base import DfFilterBase


#======================================================#
class RarePurchaseFilter(DfFilterBase):
    """
    Filters out items that fall below a minimum purchase count threshold.
    Removes rows where itemPurchaseCount_raw is less than or equal to the threshold.
    """

    purchaseCountCol: str = "itemTotalPurchCountToDate_raw"

    minPurchaseThreshold: int
    logger: logging.Logger

    #======================================================#
    def __init__(self, minPurchaseThreshold: int = 1):
        """
        :param minPurchaseThreshold: Minimum purchase count required to keep an item.
            Rows at or below this value will be removed. Defaults to 1.
        :type minPurchaseThreshold: int
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.minPurchaseThreshold = minPurchaseThreshold
        self.logger.info("RarePurchaseFilter initialized minPurchaseThreshold=%s", self.minPurchaseThreshold)

    #======================================================#
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows where itemPurchaseCount_raw is at or below the minimum threshold.

        :param df: Input DataFrame containing the itemPurchaseCount_raw column.
        :type df: pd.DataFrame
        :returns: DataFrame with rare purchase items removed.
        :rtype: pd.DataFrame
        :raises ValueError: If the required column is missing.
        """
        self.logger.info("filter(): start rows=%s threshold=%s", len(df), self.minPurchaseThreshold)

        if self.purchaseCountCol not in df.columns:
            raise ValueError(f"{self.__class__.__name__} missing required column: {self.purchaseCountCol}")

        df = df[df[self.purchaseCountCol] > self.minPurchaseThreshold].reset_index(drop=True)

        self.logger.info("filter(): done rows=%s", len(df))
        return df